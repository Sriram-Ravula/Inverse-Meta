import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

import os
import sys
import torch.utils.tensorboard as tb
import yaml
import sigpy.mri

from algorithms.ddrm import DDRM
from problems.fourier_multicoil import MulticoilForwardMRINoMask
from datasets import get_dataset, split_dataset

from utils_new.exp_utils import save_images, save_to_pickle
from utils_new.meta_utils import hessian_vector_product as hvp
from utils_new.meta_loss_utils import meta_loss, get_meta_grad
from utils_new.inner_loss_utils import get_likelihood_grad, log_cond_likelihood_loss
from utils_new.metric_utils import Metrics

class GBML:
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        #running parameters
        self._init_c()
        self._init_meta_optimizer()
        self._init_dataset()
        self.A = None

        self.global_epoch = 0
        self.c_list = [self.c.detach().clone().cpu()]

        c_shaped = self._shape_c(self.c) #properly re-shape c before giving to DDRM
        self.langevin_runner = DDRM(self.hparams, self.args, c_shaped).to(self.device)

        if self.hparams.gpu_num == -1:
            self.langevin_runner = torch.nn.DataParallel(self.langevin_runner)

        #logging and metrics
        self.metrics = Metrics(hparams=self.hparams)
        self.log_dir = os.path.join(self.hparams.save_dir, self.args.doc)
        self.image_root = os.path.join(self.log_dir, 'images')
        self.tb_root = os.path.join(self.log_dir, 'tensorboard')

        self._make_log_folder()
        self._save_config()

        self.tb_logger = tb.SummaryWriter(log_dir=self.tb_root)

        #take a snap of the initialization
        if not self.hparams.debug and self.hparams.save_imgs:
            c_shaped = torch.abs(self._shape_c(self.c))
            c_shaped_binary = torch.zeros_like(c_shaped)
            c_shaped_binary[c_shaped > 0] = 1

            c_path = os.path.join(self.image_root, "learned_masks")

            c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu()])
            self._add_tb_images(c_out, "Mask Initialization")
            if not os.path.exists(c_path):
                os.makedirs(c_path)
            self._save_images(c_out, ["Actual_00", "Binary_00"], c_path)

    def run_meta_opt(self):
        for iter in tqdm(range(self.hparams.opt.num_iters)):
            #checkpoint
            if iter % self.hparams.opt.checkpoint_iters == 0:
                self._checkpoint()

            #validate
            if iter % self.hparams.opt.val_iters == 0:
                self._run_validation()
                self._add_metrics_to_tb("val")

            #train
            self._run_outer_step()
            self._add_metrics_to_tb("train")

            self.global_epoch += 1

        #test
        self._run_test()
        self._add_metrics_to_tb("test")

        self._checkpoint()

    def _run_outer_step(self):
        """
        Runs one epoch of meta learning training.
        """
        self._print_if_verbose("\nTRAINING\n")

        meta_grad = 0.0
        n_samples = 0
        num_batches = self.hparams.opt.batches_per_iter
        if num_batches < 0 or num_batches > len(self.train_loader):
            num_batches = len(self.train_loader)
        elif num_batches == 0:
            return

        for i, (item, x_idx) in tqdm(enumerate(self.train_loader)):
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "train")

            #(2) Calculate the meta-gradient and (possibly) update
            meta_grad += self._mle_grad(x_hat, x, y)

            #if we have passed through the requisite number of samples, update
            n_samples += x.shape[0]
            num_batches -= 1
            if num_batches==0:
                meta_grad /= n_samples
                self._opt_step(meta_grad)

                #put in the proper shape before giving to DDRM (ensures proper singular vals)
                c_shaped = self._shape_c(self.c) 
                if self.hparams.gpu_num != -1:
                    self.langevin_runner.set_c(c_shaped)
                else:
                    self.langevin_runner.module.set_c(c_shaped)

                meta_grad = 0.0
                n_samples = 0
                num_batches = self.hparams.opt.batches_per_iter
                if num_batches < 0 or num_batches > len(self.train_loader):
                    num_batches = len(self.train_loader)

            #logging and saving
            if i == 0:
                self._save_all_images(x_hat, x, y, x_idx, "train")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "train", False)
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "train"), "\n")

    def _run_validation(self):
        self._print_if_verbose("\nVALIDATING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.val_loader)):
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "val")

            #logging and saving
            if i == 0:
                self._save_all_images(x_hat, x, y, x_idx, "val")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "val", False)
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "val"), "\n")

    def _run_test(self):
        self._print_if_verbose("\nTESTING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")

            #logging and saving
            self._save_all_images(x_hat, x, y, x_idx, "test")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test", False)
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")

    def _shared_step(self, item):
        x = item['gt_image'].to(self.device) #[N, 2, H, W] float second channel is (Re, Im)
        y = item['ksp'].type(torch.cfloat).to(self.device) #[N, C, H, W, 2] float last channel is ""
        s_maps = item['s_maps'].to(self.device) #[N, C, H, W] complex

        #set coil maps and forward operator including current coil maps
        self.langevin_runner.module.H_funcs.s_maps = s_maps
        self.A = lambda x: MulticoilForwardMRINoMask()(torch.complex(x[:,0], x[:,1]), s_maps)

        #Get the reconstruction and log batch metrics
        x_mod = torch.rand_like(x)
        x_hat = self.langevin_runner(x_mod, y)

        #Y (k-space meas) is acting weirdly - comes in as 2-channel complex float
        #each channel has all real entries
        #But after Langevin, contains actual complex dtype with a single channel with (Re, Im)

        print("Image shape: ", x.shape)
        print("Image type: ", x.dtype)
        print("Recon shape: ", x_hat.shape)
        print("Recon type: ", x_hat.dtype)
        print("K-Space shape: ", y.shape)
        print("K-Space dtype: ", y.dtype)
        print("Coil map shape: ", s_maps.shape)
        print("Coil map dtype: ", s_maps.dtype)

        return x_hat, x, y

    @torch.no_grad()
    def _add_batch_metrics(self, x_hat, x, y, iter_type):
        """
        Adds metrics for a single batch to the metrics object.
        """
        real_meas_loss = log_cond_likelihood_loss(torch.tensor(1.), y, self.A, x_hat, 2., reduce_dims=tuple(np.arange(y.dim())[1:])) #get element-wise MSE
        c_shaped = self._shape_c(self.c)
        weighted_meas_loss = log_cond_likelihood_loss(c_shaped, y, self.A, x_hat, 2.,
                                                      reduce_dims=tuple(np.arange(y.dim())[1:])) #get element-wise MSE with mask
        all_meta_losses = meta_loss(x_hat, x, tuple(np.arange(x.dim())[1:]), self.c,
                                    meta_loss_type=self.hparams.outer.meta_loss_type,
                                    reg_hyperparam=self.hparams.outer.reg_hyperparam,
                                    reg_hyperparam_type=self.hparams.outer.reg_hyperparam_type,
                                    reg_hyperparam_scale=self.hparams.outer.reg_hyperparam_scale)

        extra_metrics_dict = {"real_meas_loss": torch.abs(real_meas_loss).cpu().numpy().flatten(),
                              "weighted_meas_loss": torch.abs(weighted_meas_loss).cpu().numpy().flatten(),
                              "meta_loss_"+str(self.hparams.outer.meta_loss_type): all_meta_losses[0].cpu().numpy().flatten(),
                              "meta_loss_reg": all_meta_losses[1].cpu().numpy().flatten(),
                              "meta_loss_total": all_meta_losses[2].cpu().numpy().flatten()}

        if self.hparams.outer.reg_hyperparam:
            sparsity_level = 1 - (self.c.count_nonzero() / self.c.numel())
            extra_metrics_dict["sparsity_level"] = np.array([sparsity_level.item()] * x.shape[0]) #ugly artifact
            extra_metrics_dict["c_min"] = np.array([torch.min(self.c).item()] * x.shape[0])
            extra_metrics_dict["c_max"] = np.array([torch.max(self.c).item()] * x.shape[0])
            extra_metrics_dict["c_mean"] = np.array([torch.mean(self.c).item()] * x.shape[0])
            extra_metrics_dict["c_std"] = np.array([torch.std(self.c).item()] * x.shape[0])

            self._add_histogram_to_tb(self.c, "C values")

        self.metrics.add_external_metrics(extra_metrics_dict, self.global_epoch, iter_type)
        self.metrics.calc_iter_metrics(x_hat, x, self.global_epoch, iter_type)

    @torch.no_grad()
    def _save_all_images(self, x_hat, x, y, x_idx, iter_type):
        """
        Given true, measurement, and recovered images, save to tensorboard and png.
        """
        if self.hparams.debug or (not self.hparams.save_imgs):
            return
        
        #(1) Save samping masks
        if iter_type == "train":
            c_shaped = torch.abs(self._shape_c(self.c))
            c_shaped_binary = torch.zeros_like(c_shaped)
            c_shaped_binary[c_shaped > 0] = 1

            c_path = os.path.join(self.image_root, "learned_masks")

            c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu()])
            self._add_tb_images(c_out, "Learned Mask")
            if not os.path.exists(c_path):
                os.makedirs(c_path)
            self._save_images(c_out, ["Actual_" + str(self.global_epoch),
                                      "Binary_" + str(self.global_epoch)], c_path)

        #(2) Save reconstructions at every iteration
        meas_recovered_path = os.path.join(self.image_root, iter_type + "_recon_meas", "epoch_"+str(self.global_epoch))
        recovered_path = os.path.join(self.image_root, iter_type + "_recon", "epoch_"+str(self.global_epoch))

        x_hat_vis = torch.norm(x_hat, dim=1).unsqueeze(1) #[N, 1, H, W]

        self._add_tb_images(x_hat_vis, "recovered " + iter_type + " images")
        if not os.path.exists(recovered_path):
            os.makedirs(recovered_path)
        self._save_images(x_hat_vis, x_idx, recovered_path)
        
        fake_maps = torch.ones_like(x)[:,0,:,:].unsqueeze(1) #[N, 1, H, W]
        recon_meas = MulticoilForwardMRINoMask()(torch.complex(x_hat[:,0], x_hat[:,1]), fake_maps)
        recon_meas = torch.abs(recon_meas)

        self._add_tb_images(recon_meas, "recovered " + iter_type + " meas")
        if not os.path.exists(meas_recovered_path):
            os.makedirs(meas_recovered_path)
        self._save_images(recon_meas, x_idx, meas_recovered_path)

        #(3) Save ground truth only once
        if iter_type == "test" or self.global_epoch == 0:
            true_path = os.path.join(self.image_root, iter_type)
            meas_path = os.path.join(self.image_root, iter_type + "_meas")

            x_vis = torch.norm(x, dim=1).unsqueeze(1) #[N, 1, H, W]

            self._add_tb_images(x_vis, iter_type + " images")
            if not os.path.exists(true_path):
                os.makedirs(true_path)
            self._save_images(x_vis, x_idx, true_path)

            gt_meas = MulticoilForwardMRINoMask()(torch.complex(x[:,0], x[:,1]), fake_maps)
            gt_meas = torch.abs(gt_meas)

            self._add_tb_images(gt_meas, iter_type + " meas")
            if not os.path.exists(meas_path):
                os.makedirs(meas_path)
            self._save_images(gt_meas, x_idx, meas_path)

    def _opt_step(self, meta_grad):
        """
        Will take an optimization step (and scheduler if applicable).
        Sets c.grad to True then False.
        """
        self.opt.zero_grad()
        self.c.requires_grad_()

        #dummy update to make sure grad is initialized
        if type(self.c.grad) == type(None):
            dummy_loss = torch.sum(self.c)
            dummy_loss.backward()
        
        self.c.grad.copy_(meta_grad)
        self.opt.step()
        self.c.requires_grad_(False)

        if self.hparams.outer.reg_hyperparam:
            #the scale parameter in soft is lambda for soft thresholding with the 
            #   proximal l1-sparsity operator
            if self.hparams.outer.reg_hyperparam_type == "soft":
                over_idx = (self.c > self.hparams.outer.reg_hyperparam_scale)
                mid_idx = (self.c >= -self.hparams.outer.reg_hyperparam_scale) & \
                            (self.c <= self.hparams.outer.reg_hyperparam_scale)
                under_idx = (self.c < -self.hparams.outer.reg_hyperparam_scale)
                self.c[over_idx] -= self.hparams.outer.reg_hyperparam_scale
                self.c[mid_idx] *= 0
                self.c[under_idx] += self.hparams.outer.reg_hyperparam_scale
            
            #the scale parameter in hard thresholding means keep <scale> highest values
            #   and zero the (1 - <scale>) remaining 
            elif self.hparams.outer.reg_hyperparam_type == "hard":
                k = int(self.c.numel() * (1 - self.hparams.outer.reg_hyperparam_scale))
                smallest_kept_val = torch.kthvalue(torch.abs(self.c), k)[0]
                under_idx = torch.abs(self.c) < smallest_kept_val
                self.c[under_idx] *= 0
            
            elif self.hparams.outer.reg_hyperparam_type != "l1":
                raise NotImplementedError("This meta regularizer has not been implemented yet!")
            
            self.c.clamp_(min=0., max=1.) #TODO check if clamping at 0 or -1 is better
        
        self.c_list.append(self.c.detach().clone().cpu())

        if self.scheduler is not None and self.hparams.opt.decay:
            LR_OLD = self.opt.param_groups[0]['lr']
            self.scheduler.step()
            LR_NEW = self.opt.param_groups[0]['lr']
            self._print_if_verbose("\nDECAYING LR: ", LR_OLD, " --> ", LR_NEW)

    def _mle_grad(self, x_hat, x, y):
        """
        Calculates the meta-gradient for an MLE step.
        grad_c(meta_loss) = - grad_x_c[recon_loss] * (meta_loss)
        (1) Find meta loss
        (2) Get the HVP grad_x_c(recon_loss) * grad_x(meta_loss)

        Sets c.grad to True then False.
        """
        #(1) Get gradients of Meta loss w.r.t. image and hyperparams
        grad_x_meta_loss, grad_c_meta_loss = get_meta_grad(x_hat=x_hat,
                                                            x_true=x,
                                                            c = self.c,
                                                            meta_loss_type=self.hparams.outer.meta_loss_type,
                                                            reg_hyperparam=self.hparams.outer.reg_hyperparam,
                                                            reg_hyperparam_type=self.hparams.outer.reg_hyperparam_type,
                                                            reg_hyperparam_scale=self.hparams.outer.reg_hyperparam_scale)

        #(2)
        self.c.requires_grad_()

        c_shaped = self._shape_c(self.c)
        cond_log_grad = get_likelihood_grad(c_shaped, y, self.A, x_hat,
                                            retain_graph=True,
                                            create_graph=True)
        out_grad = 0.0
        out_grad -= hvp(self.c, cond_log_grad, grad_x_meta_loss)
        out_grad += grad_c_meta_loss

        self.c.requires_grad_(False)

        return out_grad

    def _init_dataset(self):
        _, base_dataset = get_dataset(self.hparams)
        split_dict = split_dataset(base_dataset, self.hparams)
        train_dataset = split_dict['train']
        val_dataset = split_dict['val']
        test_dataset = split_dict['test']

        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.data.train_batch_size, shuffle=True,
                                num_workers=1, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=1, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.hparams.data.test_batch_size, shuffle=False,
                                num_workers=1, drop_last=True)
    
    def _shape_c(self, c):
        """
        Function for properly shaping c to broadcast with images and measurements
        """
        if self.hparams.problem.measurement_weighting:
            c_shaped = torch.ones((self.hparams.data.image_size, self.hparams.data.image_size)) * c

        elif self.hparams.problem.measurement_selection:
            if self.hparams.problem.sample_pattern == 'random':
                c_shaped = c.view(self.hparams.data.image_size, self.hparams.data.image_size)

            elif self.hparams.problem.sample_pattern == 'horizontal':
                c_shaped = c.unsqueeze(1).repeat(1, self.hparams.data.image_size)

            elif self.hparams.problem.sample_pattern == 'vertical':
                c_shaped = c.unsqueeze(0).repeat(self.hparams.data.image_size, 1)
            
            else:
                raise NotImplementedError("This sample pattern is not supported!")
            
        return c_shaped

    def _init_c(self):
        """
        Initializes C.
        (1) Check for measurement_selection; False means c scalar, and we are done
        (2) Check for sample_pattern; resize accordingly
        (3) Check for any smart initialization
        """

        problem_check = sum([self.hparams.problem.measurement_weighting, 
                             self.hparams.problem.measurement_selection])
        assert problem_check == 1, "Must choose exactly one of measurement weighting and selection!"

        if self.hparams.problem.measurement_weighting:
            c = torch.tensor(1.)
            
        elif self.hparams.problem.measurement_selection:
            if self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                m = self.hparams.data.image_size

            elif self.hparams.problem.sample_pattern == 'random':
                m = self.hparams.data.image_size ** 2

            else:
                raise NotImplementedError("Fourier sampling pattern not supported!")

            #now check for any smart initializations
            if self.hparams.outer.hyperparam_init == "random":
                c = torch.rand(m)

            elif isinstance(self.hparams.outer.hyperparam_init, (int, float)):
                c = torch.ones(m) * float(self.hparams.outer.hyperparam_init)

            elif self.hparams.outer.hyperparam_init == "smart":
                if self.hparams.problem.sample_pattern == 'random':
                    c = sigpy.mri.poisson(img_shape=(self.hparams.data.image_size, self.hparams.data.image_size),
                                          accel=self.hparams.problem.R,
                                          seed=self.hparams.seed)
                    c = torch.tensor(c)
                    c = torch.view_as_real(c)[:,:,0].type(torch.float)
                    c = c.flatten() 

                elif self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                    num_sampled_lines = np.floor(self.hparams.data.image_size / self.hparams.problem.R)
                    center_line_idx = np.arange((self.hparams.data.image_size - 30) // 2,
                                        (self.hparams.data.image_size + 30) // 2)
                    outer_line_idx = np.setdiff1d(np.arange(self.hparams.data.image_size), center_line_idx)
                    random_line_idx = outer_line_idx[::int(self.hparams.problem.R)]
                    c = torch.zeros(self.hparams.data.image_size)
                    c[center_line_idx] = 1.
                    c[random_line_idx] = 1. 
            
        self.c = c.to(self.device)
        return

    def _init_meta_optimizer(self):
        """
        Initializes the meta optmizer and scheduler.
        """

        opt_type = self.hparams.opt.optimizer
        lr = self.hparams.opt.lr

        if opt_type == 'adam':
            meta_opt = torch.optim.Adam([{'params': self.c}], lr=lr)
        elif opt_type == 'sgd':
            meta_opt = torch.optim.SGD([{'params': self.c}], lr=lr)
        else:
            raise NotImplementedError("Optimizer not supported!")

        if self.hparams.opt.decay:
            meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_opt, self.hparams.opt.lr_decay)
        else:
            meta_scheduler = None

        self.opt =  meta_opt
        self.scheduler = meta_scheduler

    def _checkpoint(self):
        if not self.hparams.debug:
            save_dict = {
                "c": self.c,
                "c_list": self.c_list,
                "global_epoch": self.global_epoch,
                "opt_state": self.opt.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None
            }
            metrics_dict = {
                'train_metrics': self.metrics.train_metrics,
                'val_metrics': self.metrics.val_metrics,
                'test_metrics': self.metrics.test_metrics,
                'train_metrics_aggregate': self.metrics.train_metrics_aggregate,
                'val_metrics_aggregate': self.metrics.val_metrics_aggregate,
                'test_metrics_aggregate': self.metrics.test_metrics_aggregate,
                'best_train_metrics': self.metrics.best_train_metrics,
                'best_val_metrics': self.metrics.best_val_metrics,
                'best_test_metrics': self.metrics.best_test_metrics
            }
            save_to_pickle(save_dict, os.path.join(self.log_dir, "checkpoint.pkl"))
            save_to_pickle(metrics_dict, os.path.join(self.log_dir, "metrics.pkl"))

    def _make_log_folder(self):
        if not self.hparams.debug:
            if os.path.exists(self.log_dir):
                sys.exit("Folder exists. Program halted.")
            else:
                os.makedirs(self.log_dir)
                os.makedirs(self.image_root)
                os.makedirs(self.tb_root)

    def _save_config(self):
        if not self.hparams.debug:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as f:
                yaml.dump(self.hparams, f, default_flow_style=False)

    def _add_tb_images(self, images, tag):
        if not self.hparams.debug and self.hparams.save_imgs:
            grid_img = torchvision.utils.make_grid(images.cpu(), nrow=images.shape[0]//2)
            self.tb_logger.add_image(tag, grid_img, global_step=self.global_epoch)

    def _add_metrics_to_tb(self, iter_type):
        if not self.hparams.debug:
            self.metrics.add_metrics_to_tb(self.tb_logger, self.global_epoch, iter_type)

    def _add_histogram_to_tb(self, values, tag):
        if not self.hparams.debug:
            self.tb_logger.add_histogram(tag, values, global_step=self.global_epoch)

    def _save_images(self, images, img_indices, save_path):
        if not self.hparams.debug and self.hparams.save_imgs:
            save_images(images, img_indices, save_path)

    def _print_if_verbose(self, *text):
        if self.hparams.verbose:
            print("".join(str(t) for t in text))
