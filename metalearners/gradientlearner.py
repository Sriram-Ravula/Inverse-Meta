import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import torch.fft as torch_fft

import os
import sys
import torch.utils.tensorboard as tb
import yaml
import sigpy.mri

# from algorithms.ddrm import DDRM
from algorithms.wavelet import L1_wavelet
from algorithms.ncsnv2 import NCSNv2
from algorithms.mvue import MVUE_solution
from problems.fourier_multicoil import MulticoilForwardMRINoMask
from datasets import get_dataset, split_dataset

from utils_new.exp_utils import save_images, save_to_pickle, load_if_pickled
from utils_new.meta_utils import hessian_vector_product as hvp
from utils_new.meta_loss_utils import meta_loss
from utils_new.metric_utils import Metrics

from metalearners.probabilistic_mask import Probabilistic_Mask

class GBML:
    def __init__(self, hparams, args):
        #NOTE has been optimized for probabilistic mask

        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        if self.args.resume:
            self._resume()
            return

        #check if we have a probabilistic c
        self.prob_c = hasattr(self.hparams.outer, 'sample_pattern')
        if self.prob_c:
            self._print_if_verbose("USING PROBABILISTIC MASK!")

        #check if we have an ROI
        self.ROI = getattr(self.hparams.outer, 'ROI', None) #this should be [(H_start, H_end), (W_start, W_end)]

        #running parameters
        self._init_c()
        self._init_meta_optimizer()
        self._init_dataset()
        self.A = None #placeholder for forward operator - None since each sample has different coil map

        self.global_epoch = 0

        if self.prob_c:
            tau = getattr(self.hparams.outer, 'tau', 0.5)
            self._print_if_verbose("TAU = ", tau)
            self.c_list = [self.c.weights.detach().clone().cpu()]
            self.cur_mask_sample = self.c.sample_mask(tau) #draw a binary mask sample
            c_shaped = self.cur_mask_sample.detach().clone()
        else:
            self.c_list = [self.c.detach().clone().cpu()]
            c_shaped = self._shape_c(self.c) 

        if self.hparams.net.model == 'ncsnpp':
            self.recon_alg = DDRM(self.hparams, self.args, c_shaped, self.device).to(self.device)
        elif self.hparams.net.model == 'l1':
            self.recon_alg = L1_wavelet(self.hparams, self.args, c_shaped)
        elif self.hparams.net.model == 'ncsnv2':
            self.recon_alg = NCSNv2(self.hparams, self.args, c_shaped, self.device).to(self.device)
        elif self.hparams.net.model == 'mvue':
            self.recon_alg = MVUE_solution(self.hparams, self.args, c_shaped)

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
            if self.ROI is not None:
                H0, H1 = self.ROI[0]
                W0, W1 = self.ROI[1]

                ROI_IMG = torch.zeros(1, 1, self.hparams.data.image_size, self.hparams.data.image_size)
                ROI_IMG[..., H0:H1, W0:W1] = 1.0

                ROI_path = os.path.join(self.image_root, "learned_masks")

                if not os.path.exists(ROI_path):
                    os.makedirs(ROI_path)
                self._save_images(ROI_IMG, ["ROI"], ROI_path)

            if self.prob_c:
                c_shaped = self.c.get_prob_mask()
                c_shaped_binary = self.cur_mask_sample.detach().clone()
                c_shaped_max = self.c.get_max_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu(), c_shaped_max.unsqueeze(0).cpu()])

                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Prob_00", "Sample_00", "Max_00"], c_path)

                #NOTE sparsity level is the proportion of zeros in the image
                sparsity_level = 1 - (c_shaped_binary.count_nonzero() / c_shaped_binary.numel())
                self._print_if_verbose("INITIAL SPARSITY (SAMPLE MASK): " + str(sparsity_level.item()))

                sparsity_level = 1 - (c_shaped_max.count_nonzero() / c_shaped_max.numel())
                self._print_if_verbose("INITIAL SPARSITY (MAX MASK): " + str(sparsity_level.item()))
            else:
                c_shaped = torch.abs(self._shape_c(self.c))
                c_shaped_binary = torch.zeros_like(c_shaped)
                c_shaped_binary[c_shaped > 0] = 1

                c_path = os.path.join(self.image_root, "learned_masks")

                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu()])
                # self._add_tb_images(c_out, "Mask Initialization")
                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Actual_00", "Binary_00"], c_path)

                #NOTE sparsity level is the proportion of zeros in the image
                sparsity_level = 1 - (self.c.count_nonzero() / self.c.numel())
                self._print_if_verbose("INITIAL SPARSITY: " + str(sparsity_level.item()))

    def test(self):
        """
        Run through the test set with a given acceleration and center value.
        We want to save the metrics for each individual sample here!
        We also want to save images of every sample, reconstruction, measurement, recon_meas,
            and the c.
        NOTE NOT optimized for probabilistic mask yet
        """
        R = self.args.R
        keep_center = self.args.keep_center

        self._print_if_verbose("TESTING R="+str(R)+", KEEP CENTER="+str(keep_center))

        #make c the right acceleration and sample center if needed
        c = self.c.detach().clone()
        if R > 1:
            k = int(c.numel() * (1. - 1. / R))
            smallest_kept_val = torch.kthvalue(torch.abs(c), k)[0]
            under_idx = torch.abs(c) < smallest_kept_val
            c[under_idx] = 0.
        if keep_center:
            num_center_lines = getattr(self.hparams.problem, 'num_acs_lines', 20)
            # num_center_lines = int(self.hparams.data.image_size // 12) #keep ~8% of center
            center_line_idx = np.arange((self.hparams.data.image_size - num_center_lines) // 2,
                                (self.hparams.data.image_size + num_center_lines) // 2)

            if self.hparams.problem.sample_pattern == 'random':
                center_line_idx = np.meshgrid(center_line_idx, center_line_idx)
                c = c.view(self.hparams.data.image_size, self.hparams.data.image_size)
                c[center_line_idx] = 1.
                c = c.flatten()

            elif self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                c[center_line_idx] = 1.

        c[c > 0] = 1. #binarize the learned mask
        self.c = c.to(self.device)

        c_shaped = self._shape_c(self.c)
        self.recon_alg.set_c(c_shaped)

        #for logging and metrics purposes; avoids collisions with existing test
        self.global_epoch = self.hparams.opt.num_iters + R 
        if keep_center:
            self.global_epoch += 1

        #take a snap of the initialization
        if not self.hparams.debug and self.hparams.save_imgs:
            c_path = os.path.join(self.image_root, "learned_masks")

            c_out = c_shaped.unsqueeze(0).unsqueeze(0).cpu()
            if not os.path.exists(c_path):
                os.makedirs(c_path)
            self._save_images(c_out, ["TEST_R"+str(R)+"_CENTER-"+str(keep_center)], c_path)

            #NOTE sparsity level is the proportion of zeros in the image
            sparsity_level = 1 - (self.c.count_nonzero() / self.c.numel())
            self._print_if_verbose("INITIAL SPARSITY: " + str(sparsity_level.item()))

        #Test
        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")
            #logging and saving
            scan_idxs = item['scan_idx']
            slice_idxs = item['slice_idx']
            x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
            self._save_all_images(x_hat, x, y, x_idx, "test_R"+str(R)+"_CENTER-"+str(keep_center))

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test")
        self._add_metrics_to_tb("test")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")

        #grab the raw metrics dictionary and save it
        test_metrics = self.metrics.test_metrics['iter_'+str(self.global_epoch)]
        save_to_pickle(test_metrics, os.path.join(self.log_dir, "test_R"+str(R)+"_CENTER-"+str(keep_center)+".pkl"))

        return

    def run_meta_opt(self):
        #NOTE doesn't need optimization for probabilistic mask
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
        NOTE has been optimized for probabilistic C
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

                if self.prob_c:
                    tau = getattr(self.hparams.outer, 'tau', 0.5)
                    self.cur_mask_sample = self.c.sample_mask(tau)
                    c_shaped = self.cur_mask_sample.detach().clone()
                else:
                    c_shaped = self._shape_c(self.c)
                self.recon_alg.set_c(c_shaped)

                meta_grad = 0.0
                n_samples = 0
                num_batches = self.hparams.opt.batches_per_iter
                if num_batches < 0 or num_batches > len(self.train_loader):
                    num_batches = len(self.train_loader)

            #logging and saving
            if i == 0:
                scan_idxs = item['scan_idx']
                slice_idxs = item['slice_idx']
                x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
                self._save_all_images(x_hat, x, y, x_idx, "train")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "train")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "train"), "\n")

    def _run_validation(self):
        #NOTE doesn't need optimization for probabilistic mask
        self._print_if_verbose("\nVALIDATING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.val_loader)):
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "val")

            #draw a new sample for every validation image
            if self.prob_c:
                tau = getattr(self.hparams.outer, 'tau', 0.5)
                self.cur_mask_sample = self.c.sample_mask(tau)
                c_shaped = self.cur_mask_sample.detach().clone()
            else:
                c_shaped = self._shape_c(self.c)
            self.recon_alg.set_c(c_shaped)

            #logging and saving
            if i == 0:
                scan_idxs = item['scan_idx']
                slice_idxs = item['slice_idx']
                x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
                self._save_all_images(x_hat, x, y, x_idx, "val")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "val")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "val"), "\n")

    def _run_test(self):
        #NOTE doesn't need optimization for probabilistic mask
        self._print_if_verbose("\nTESTING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")

            #draw a new sample for every test image
            if self.prob_c:
                tau = getattr(self.hparams.outer, 'tau', 0.5)
                self.cur_mask_sample = self.c.sample_mask(tau)
                c_shaped = self.cur_mask_sample.detach().clone()
            else:
                c_shaped = self._shape_c(self.c)
            self.recon_alg.set_c(c_shaped)

            #logging and saving
            scan_idxs = item['scan_idx']
            slice_idxs = item['slice_idx']
            x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
            self._save_all_images(x_hat, x, y, x_idx, "test")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")

    def _shared_step(self, item):
        #NOTE doesn't need optimization for probabilistic mask
        x = item['gt_image'].to(self.device) #[N, 2, H, W] float second channel is (Re, Im)
        y = item['ksp'].type(torch.cfloat).to(self.device) #[N, C, H, W, 2] float last channel is ""
        s_maps = item['s_maps'].to(self.device) #[N, C, H, W] complex
        scale_factor = item['scale_factor'].to(self.device)

        #set coil maps and forward operator including current coil maps
        self.recon_alg.H_funcs.s_maps = s_maps
        self.A = MulticoilForwardMRINoMask(s_maps)

        #Get the reconstruction
        x_mod = torch.randn_like(x)
        x_hat = self.recon_alg(x_mod, y) #[N, 2, H, W] float

        #scale the output appropriately
        x_hat_scale_ = np.percentile(np.linalg.norm(x_hat.view(x_hat.shape[0],x_hat.shape[1],-1).detach().cpu().numpy(), axis=1), 99, axis=1)
        x_hat_scale = torch.Tensor(x_hat_scale_).to(self.device)
        x_hat /= x_hat_scale[:, None, None, None]

        #Do a fully-sampled forward-->adjoint on the output 
        x_hat = self.A(x_hat) #[N, C, H, W] complex in kspace domain
        x_hat = torch.view_as_real(torch.sum(self._ifft(x_hat) * torch.conj(s_maps), axis=1) ).permute(0,3,1,2)

        print("Image shape: ", x.shape)
        print("Image type: ", x.dtype)
        print("Recon shape: ", x_hat.shape)
        print("Recon type: ", x_hat.dtype)
        print("K-Space shape: ", y.shape)
        print("K-Space dtype: ", y.dtype)
        print("Coil map shape: ", s_maps.shape)
        print("Coil map dtype: ", s_maps.dtype)

        return x_hat, x, y

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x 

    @torch.no_grad()
    def _add_batch_metrics(self, x_hat, x, y, iter_type):
        """
        Adds metrics for a single batch to the metrics object.
        NOTE optimized for probabilistic C
        """
        resid = self.A(x_hat) - y

        real_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise MSE
                                                      
        if self.prob_c:
            c_shaped = self.cur_mask_sample.detach().clone()
            resid = c_shaped[None, None, :, :] * resid
            weighted_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise MSE with mask

            all_meta_losses = meta_loss(x_hat, x, tuple(np.arange(x.dim())[1:]), c_shaped,
                                        meta_loss_type=self.hparams.outer.meta_loss_type,
                                        reg_hyperparam=False,
                                        reg_hyperparam_type=None,
                                        reg_hyperparam_scale=0)
        else:
            c_shaped = self._shape_c(self.c)
            resid = c_shaped[None, None, :, :] * resid

            weighted_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise MSE with mask

            all_meta_losses = meta_loss(x_hat, x, tuple(np.arange(x.dim())[1:]), self.c,
                                        meta_loss_type=self.hparams.outer.meta_loss_type,
                                        reg_hyperparam=self.hparams.outer.reg_hyperparam,
                                        reg_hyperparam_type=self.hparams.outer.reg_hyperparam_type,
                                        reg_hyperparam_scale=self.hparams.outer.reg_hyperparam_scale)

        extra_metrics_dict = {"real_meas_loss": real_meas_loss.cpu().numpy().flatten(),
                            "weighted_meas_loss": weighted_meas_loss.cpu().numpy().flatten(),
                            "meta_loss_"+str(self.hparams.outer.meta_loss_type): all_meta_losses[0].cpu().numpy().flatten(),
                            "meta_loss_reg": all_meta_losses[1].cpu().numpy().flatten(),
                            "meta_loss_total": all_meta_losses[2].cpu().numpy().flatten()}
        
        if self.prob_c:
            prob_mask = self.c.get_prob_mask()
            extra_metrics_dict["mean_prob"] = np.array([torch.mean(prob_mask).item()] * x.shape[0])

            max_mask = self.c.get_max_mask()
            sparsity_level_max = 1 - (max_mask.count_nonzero() / max_mask.numel())
            extra_metrics_dict["sparsity_level_max"] = np.array([sparsity_level_max.item()] * x.shape[0]) #ugly artifact

            cur_mask = self.cur_mask_sample.detach().clone()
            sparsity_level_sample = 1 - (cur_mask.count_nonzero() / cur_mask.numel())
            extra_metrics_dict["sparsity_level_sample"] = np.array([sparsity_level_sample.item()] * x.shape[0]) #ugly artifact
        else:
            if self.hparams.outer.reg_hyperparam:
                sparsity_level = 1 - (self.c.count_nonzero() / self.c.numel())
                extra_metrics_dict["sparsity_level"] = np.array([sparsity_level.item()] * x.shape[0]) #ugly artifact
                extra_metrics_dict["c_min"] = np.array([torch.min(self.c).item()] * x.shape[0])
                extra_metrics_dict["c_max"] = np.array([torch.max(self.c).item()] * x.shape[0])
                extra_metrics_dict["c_mean"] = np.array([torch.mean(self.c).item()] * x.shape[0])
                extra_metrics_dict["c_std"] = np.array([torch.std(self.c).item()] * x.shape[0])

        self.metrics.add_external_metrics(extra_metrics_dict, self.global_epoch, iter_type)
        self.metrics.calc_iter_metrics(x_hat, x, self.global_epoch, iter_type, self.ROI)

    @torch.no_grad()
    def _save_all_images(self, x_hat, x, y, x_idx, iter_type):
        """
        Given true, measurement, and recovered images, save to tensorboard and png.
        #NOTE optimized for probabilistic C
        """
        if self.hparams.debug or (not self.hparams.save_imgs):
            return
        elif iter_type == "train" and self.global_epoch % self.hparams.opt.checkpoint_iters != 0:
            return

        #(1) Save samping masks
        if iter_type == "train":
            if self.prob_c:
                c_shaped = self.c.get_prob_mask()
                c_shaped_binary = self.cur_mask_sample.detach().clone()
                c_shaped_max = self.c.get_max_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu(), c_shaped_max.unsqueeze(0).cpu()])

                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Prob_" + str(self.global_epoch), 
                                          "Sample_" + str(self.global_epoch), 
                                          "Max_" + str(self.global_epoch)], c_path)
            else:
                c_shaped = torch.abs(self._shape_c(self.c))
                c_shaped_binary = torch.zeros_like(c_shaped)
                c_shaped_binary[c_shaped > 0] = 1

                c_path = os.path.join(self.image_root, "learned_masks")

                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu()])
                # self._add_tb_images(c_out, "Learned Mask")
                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Actual_" + str(self.global_epoch),
                                        "Binary_" + str(self.global_epoch)], c_path)

        #(2) Save reconstructions at every iteration
        meas_recovered_path = os.path.join(self.image_root, iter_type + "_recon_meas", "epoch_"+str(self.global_epoch))
        recovered_path = os.path.join(self.image_root, iter_type + "_recon", "epoch_"+str(self.global_epoch))

        x_hat_vis = torch.norm(x_hat, dim=1).unsqueeze(1) #[N, 1, H, W]

        # self._add_tb_images(x_hat_vis, "recovered " + iter_type + " images")
        if not os.path.exists(recovered_path):
            os.makedirs(recovered_path)
        self._save_images(x_hat_vis, x_idx, recovered_path)

        if self.ROI is not None:
            H0, H1 = self.ROI[0]
            W0, W1 = self.ROI[1]

            x_hat_ROI = x_hat_vis[..., H0:H1, W0:W1]
            x_idx_ROI = [s + "_ROI" for s in x_idx]

            self._save_images(x_hat_ROI, x_idx_ROI, recovered_path)

        fake_maps = torch.ones_like(x)[:,0,:,:].unsqueeze(1) #[N, 1, H, W]
        recon_meas = MulticoilForwardMRINoMask(fake_maps)(x_hat)
        recon_meas = torch.abs(recon_meas)

        # self._add_tb_images(recon_meas, "recovered " + iter_type + " meas")
        if not os.path.exists(meas_recovered_path):
            os.makedirs(meas_recovered_path)
        self._save_images(recon_meas, x_idx, meas_recovered_path)

        #(3) Save ground truth only once
        if "test" in iter_type or self.global_epoch == 0:
            true_path = os.path.join(self.image_root, iter_type)
            meas_path = os.path.join(self.image_root, iter_type + "_meas")

            x_vis = torch.norm(x, dim=1).unsqueeze(1) #[N, 1, H, W]

            # self._add_tb_images(x_vis, iter_type + " images")
            if not os.path.exists(true_path):
                os.makedirs(true_path)
            self._save_images(x_vis, x_idx, true_path)

            if self.ROI is not None:
                H0, H1 = self.ROI[0]
                W0, W1 = self.ROI[1]

                x_ROI = x_vis[..., H0:H1, W0:W1]
                x_idx_ROI = [s + "_ROI" for s in x_idx]

                self._save_images(x_ROI, x_idx_ROI, true_path)

            gt_meas = MulticoilForwardMRINoMask(fake_maps)(x)
            gt_meas = torch.abs(gt_meas)

            # self._add_tb_images(gt_meas, iter_type + " meas")
            if not os.path.exists(meas_path):
                os.makedirs(meas_path)
            self._save_images(gt_meas, x_idx, meas_path)

    def _opt_step(self, meta_grad):
        """
        Will take an optimization step (and scheduler if applicable).
        Sets c.grad to True then False.
        NOTE optimized for probabilistic C
        """
        if self.prob_c:
            self.opt.zero_grad()

            # dummy update to make sure grad is initialized
            if type(self.c.weights.grad) == type(None):
                dummy_loss = torch.sum(self.c.weights)
                dummy_loss.backward()
            
            self.c.weights.grad.copy_(meta_grad)
            self.opt.step()

            self.c_list.append(self.c.weights.detach().clone().cpu())
        else:
            self.opt.zero_grad()
            self.c.requires_grad_()

            # dummy update to make sure grad is initialized
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

            #finally check to see if we want to keep the center
            if self.hparams.problem.measurement_selection and self.hparams.outer.keep_center:
                num_center_lines = getattr(self.hparams.problem, 'num_acs_lines', 20)
                # num_center_lines = int(self.hparams.data.image_size // 12) #keep ~8% of center
                center_line_idx = np.arange((self.hparams.data.image_size - num_center_lines) // 2,
                                    (self.hparams.data.image_size + num_center_lines) // 2)

                if self.hparams.problem.sample_pattern == 'random':
                    center_line_idx = np.meshgrid(center_line_idx, center_line_idx)
                    self.c = self.c.view(self.hparams.data.image_size, self.hparams.data.image_size)
                    self.c[center_line_idx] = 1.
                    self.c = self.c.flatten()

                elif self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                    self.c[center_line_idx] = 1.

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
        NOTE optimized for probabilistic c
        """
        #(1) Get gradients of Meta loss w.r.t. image and hyperparams
        grad_x_meta_loss = x_hat - x

        if self.ROI is not None:
            H0, H1 = self.ROI[0]
            W0, W1 = self.ROI[1]
            grad_x_meta_loss = grad_x_meta_loss[..., H0:H1, W0:W1]

        if self.prob_c:
            grad_c_meta_loss = torch.zeros_like(self.c.weights)

            #(2)
            resid = self.cur_mask_sample[None, None, :, :] * (self.A(x_hat) - y)
            cond_log_grad = torch.view_as_real(torch.sum(self.A.ifft(resid) * torch.conj(self.A.s_maps), axis=1) ).permute(0,3,1,2)

            out_grad = 0.0
            out_grad -= hvp(self.c.weights, cond_log_grad, grad_x_meta_loss)
            out_grad += grad_c_meta_loss
        else:
            grad_c_meta_loss = torch.sign(self.c) if self.hparams.outer.reg_hyperparam_type == 'l1' else torch.zeros_like(self.c)
            grad_c_meta_loss *= self.hparams.outer.reg_hyperparam_scale

            #(2)
            self.c.requires_grad_()

            c_shaped = self._shape_c(self.c)

            resid = c_shaped[None, None, :, :] * (self.A(x_hat) - y)
            cond_log_grad = torch.view_as_real(torch.sum(self.A.ifft(resid) * torch.conj(self.A.s_maps), axis=1) ).permute(0,3,1,2)

            out_grad = 0.0
            out_grad -= hvp(self.c, cond_log_grad, grad_x_meta_loss)
            out_grad += grad_c_meta_loss

            self.c.requires_grad_(False)

        #Log the metrics for each gradient
        with torch.no_grad():
            grad_metrics_dict = {"total_grad_norm": np.array([torch.norm(out_grad).item()] * x.shape[0]),
                                "meta_loss_grad_norm": np.array([torch.norm(grad_x_meta_loss).item()] * x.shape[0]),
                                "meta_reg_grad_norm": np.array([torch.norm(grad_c_meta_loss).item()] * x.shape[0]),
                                "inner_grad_norm": np.array([torch.norm(cond_log_grad).item()] * x.shape[0])}
            self.metrics.add_external_metrics(grad_metrics_dict, self.global_epoch, "train")

        return out_grad

    def _init_dataset(self):
        train_set, test_set = get_dataset(self.hparams)
        split_dict = split_dataset(train_set, test_set, self.hparams)
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

        NOTE has been updated to work with probabilistic mask
        """
        if self.prob_c:
            num_acs_lines = getattr(self.hparams.outer, 'num_acs_lines', 20)
            self._print_if_verbose("NUMBER OF ACS LINES = ", num_acs_lines)
            self.c = Probabilistic_Mask(self.hparams, self.device, num_acs_lines)
            return

        problem_check = sum([self.hparams.problem.measurement_weighting,
                             self.hparams.problem.measurement_selection])
        assert problem_check == 1, "Must choose exactly one of measurement weighting and selection!"

        if self.hparams.problem.measurement_weighting:
            c = torch.tensor(1.)

        elif self.hparams.problem.measurement_selection:
            #define the number of parameters
            if self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                m = self.hparams.data.image_size

            elif self.hparams.problem.sample_pattern == 'random':
                m = self.hparams.data.image_size ** 2

            else:
                raise NotImplementedError("Fourier sampling pattern not supported!")

            #now check for any smart initializations
            if self.hparams.outer.hyperparam_init == "random":
                c = torch.rand(m)
                num_trash_inds = int(m * (1. - 1./self.hparams.problem.R))
                trash_inds = np.random.choice(a=m, size=num_trash_inds, replace=False)
                c[trash_inds] = 0.

            elif isinstance(self.hparams.outer.hyperparam_init, (int, float)):
                c = torch.ones(m) * float(self.hparams.outer.hyperparam_init)
                num_trash_inds = int(m * (1. - 1./self.hparams.problem.R))
                trash_inds = np.random.choice(a=m, size=num_trash_inds, replace=False)
                c[trash_inds] = 0.

            elif "smart" in self.hparams.outer.hyperparam_init:
                if self.hparams.problem.sample_pattern == 'random':
                    c = sigpy.mri.poisson(img_shape=(self.hparams.data.image_size, self.hparams.data.image_size),
                                          accel=self.hparams.problem.R,
                                          seed=self.hparams.seed-1) #NOTE function errors with seg fault for seed 2023. Using -1 to fix
                    c = torch.tensor(c)
                    c = torch.view_as_real(c)[:,:,0]
                    c = c.flatten()

                elif self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                    num_center_lines = getattr(self.hparams.problem, 'num_acs_lines', 20)
                    # num_center_lines = int(self.hparams.data.image_size // 12) #keep ~8% of center
                    center_line_idx = np.arange((self.hparams.data.image_size - num_center_lines) // 2,
                                        (self.hparams.data.image_size + num_center_lines) // 2)
                    outer_line_idx = np.setdiff1d(np.arange(self.hparams.data.image_size), center_line_idx)

                    #account for the center lines when sampling the rest of the equispaced to match proper R
                    outer_R = np.round((m - num_center_lines) / (m/self.hparams.problem.R - num_center_lines))
                    random_line_idx = outer_line_idx[::int(outer_R)]
                    c = torch.zeros(self.hparams.data.image_size)
                    c[center_line_idx] = 1.
                    c[random_line_idx] = 1.
                
                #if we want a soft operator, multiply with random
                if self.hparams.outer.hyperparam_init == "soft_smart":
                    c = c * torch.rand_like(c)

            #finally check to see if we want to keep the center
            if self.hparams.outer.keep_center:
                num_center_lines = getattr(self.hparams.problem, 'num_acs_lines', 20)
                self._print_if_verbose("NUMBER OF ACS LINES = ", num_center_lines)
                # num_center_lines = int(self.hparams.data.image_size // 12) #keep ~8% of center
                center_line_idx = np.arange((self.hparams.data.image_size - num_center_lines) // 2,
                                    (self.hparams.data.image_size + num_center_lines) // 2)

                if self.hparams.problem.sample_pattern == 'random':
                    center_line_idx = np.meshgrid(center_line_idx, center_line_idx)
                    c = c.view(self.hparams.data.image_size, self.hparams.data.image_size)
                    c[center_line_idx] = 1.
                    c = c.flatten()

                elif self.hparams.problem.sample_pattern in ['horizontal', 'vertical']:
                    c[center_line_idx] = 1.

        self.c = c.to(self.device).type(torch.float)
        return

    def _init_meta_optimizer(self):
        """
        Initializes the meta optmizer and scheduler.

        NOTE has been optimized for probabilistic mask
        """
        opt_type = self.hparams.opt.optimizer
        lr = self.hparams.opt.lr

        if self.prob_c:
            if opt_type == 'adam':
                meta_opt = torch.optim.Adam([{'params': self.c.weights}], lr=lr)
            elif opt_type == 'sgd':
                meta_opt = torch.optim.SGD([{'params': self.c.weights}], lr=lr)
            else:
                raise NotImplementedError("Optimizer not supported!")
        else:
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
        #NOTE not optimized for probabilistic c yet
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
            }
            save_to_pickle(save_dict, os.path.join(self.log_dir, "checkpoint.pkl"))
            save_to_pickle(metrics_dict, os.path.join(self.log_dir, "metrics.pkl"))

    def _resume(self):
        #NOTE not optimized for probabilistic c yet
        self._print_if_verbose("RESUMING FROM CHECKPOINT")

        self.log_dir = os.path.join(self.hparams.save_dir, self.args.doc)
        self.image_root = os.path.join(self.log_dir, 'images')
        self.tb_root = os.path.join(self.log_dir, 'tensorboard')

        checkpoint = load_if_pickled(os.path.join(self.log_dir, "checkpoint.pkl"))
        metrics = load_if_pickled(os.path.join(self.log_dir, "metrics.pkl"))

        self.c = checkpoint['c'].detach().clone().to(self.device)

        self._init_meta_optimizer()
        self.opt.load_state_dict(checkpoint['opt_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self._init_dataset()
        self.A = None

        self.global_epoch = checkpoint['global_epoch']
        self.c_list = checkpoint['c_list']

        c_shaped = self._shape_c(self.c)
        if self.hparams.net.model == 'ncsnpp':
            self.recon_alg = DDRM(self.hparams, self.args, c_shaped, self.device).to(self.device)
        elif self.hparams.net.model == 'l1':
            self.recon_alg = L1_wavelet(self.hparams, self.args, c_shaped)
        elif self.hparams.net.model == 'ncsnv2':
            self.recon_alg = NCSNv2(self.hparams, self.args, c_shaped, self.device).to(self.device)
        elif self.hparams.net.model == 'mvue':
            self.recon_alg = MVUE_solution(self.hparams, self.args, c_shaped)

        self.metrics = Metrics(hparams=self.hparams)
        self.metrics.resume(metrics)

        self.tb_logger = tb.SummaryWriter(log_dir=self.tb_root)

        self._print_if_verbose("RESUMING FROM EPOCH " + str(self.global_epoch))

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
            
            with open(os.path.join(self.log_dir, 'args.yml'), 'w') as f:
                yaml.dump(self.args, f, default_flow_style=False)

    def _add_metrics_to_tb(self, iter_type):
        if not self.hparams.debug:
            self.metrics.add_metrics_to_tb(self.tb_logger, self.global_epoch, iter_type)

    def _save_images(self, images, img_indices, save_path):
        if not self.hparams.debug and self.hparams.save_imgs:
            save_images(images, img_indices, save_path)

    def _print_if_verbose(self, *text):
        if self.hparams.verbose:
            print("".join(str(t) for t in text))
