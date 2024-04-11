import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
from tqdm import tqdm
import torch.fft as torch_fft
import json
import os
import sys
import torch.utils.tensorboard as tb
import yaml

# from algorithms.ddrm import DDRM
from algorithms.wavelet import L1_wavelet
from algorithms.dps import DPS
from algorithms.mvue import MVUE_solution
from algorithms.diffusion_cg import Diffusion_CG
from algorithms.mri_diffusion import get_noise_schedule, MRI_diffusion_sampling, make_noisy_sample, single_step_posterior_estimate

from problems.fourier_multicoil import MulticoilForwardMRINoMask
from datasets import get_dataset, split_dataset

from utils_new.exp_utils import save_images, save_to_pickle, load_if_pickled
from utils_new.metric_utils import Metrics
from utils_new.helpers import normalize, unnormalize, real_to_complex, complex_to_real, get_mvue_torch, get_min_max, ifft
from utils_new.cg import ZConjGrad, get_Aop_fun, get_cg_rhs

from metalearners.probabilistic_mask import Probabilistic_Mask
from metalearners.baseline_mask import Baseline_Mask
from metalearners.unet_mask import Network_Mask

class GBML:
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        #check if we have a probabilistic c
        self.prob_c = not(self.args.baseline)

        #running parameters
        self._init_c()
        self._init_dataset()
        self.A = None #placeholder for forward operator - None since each sample has different coil map
        
        #automatic epochs selection - NOTE this is only for 3d
        if self.hparams.opt.num_iters == -1:
            n = self.hparams.data.image_size
            R = self.hparams.mask.R
            a = getattr(self.hparams.mask, 'num_acs_lines', 16)
            updates_per_epoch = self.hparams.data.num_train / self.hparams.data.train_batch_size
            
            self.hparams.opt.num_iters = int(np.ceil(((n**2) / R - a**2) / updates_per_epoch))
            
            self._print_if_verbose("Setting Epochs to " + str(self.hparams.opt.num_iters))

        if not(self.args.baseline) and not(self.args.test):
            self._init_meta_optimizer()
        
        if self.args.mask_path is not None:
            self.c.weights.requires_grad_(False)
            
            # checkpoint = load_if_pickled(os.path.join(args.mask_path))
            # self.c.weights.copy_(checkpoint["c_weights"].to(self.device))
            checkpoint = torch.load(args.mask_path, map_location=torch.device("cpu"))
            self.c.weights.copy_(checkpoint.flatten().to(self.device))
            
            self._print_if_verbose("Restoring Mask from given path")
            # self.c.weights.requires_grad_(True)

        self.global_epoch = 0
        
        #track the best validation weights and restore them before test time
        self.best_val_psnr = 0
        self.best_val_weights = None
        self.best_val_epoch = 0

        #grab a single sampling pattern to 
        c_shaped = self._sample_and_set_mask(1, False)

        if self.hparams.net.model == 'dps':
            sys.path.append("/home/sravula/Inverse_Meta/edm") #need dnnlib and torch_utils accessible
            self.recon_alg = DPS(self.hparams, self.args, c_shaped, self.device)
        elif self.hparams.net.model == 'diffusion_cg':
            sys.path.append("/home/sravula/Inverse_Meta/edm") #need dnnlib and torch_utils accessible
            self.recon_alg = Diffusion_CG(self.hparams, self.args, c_shaped, self.device)
        elif self.hparams.net.model == 'l1':
            self.recon_alg = L1_wavelet(self.hparams, self.args, c_shaped)
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

        #We need all the stuff made before we can resume
        if self.args.resume:
            self._resume()
            return

        #take a snap of the initialization
        if not self.hparams.debug and self.hparams.save_imgs:
            if self.prob_c:
                c_shaped = self.c.get_prob_mask()
                c_shaped_binary = self.cur_mask_sample.detach().clone().squeeze()
                c_shaped_max = self.c.get_max_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu(), c_shaped_max.unsqueeze(0).cpu()])

                self._save_images(c_out, ["Prob_00", "Sample_00", "Max_00"], c_path)

                #NOTE sparsity level is the proportion of zeros in the image
                sparsity_level = 1 - (c_shaped_binary.count_nonzero() / c_shaped_binary.numel())
                self._print_if_verbose("INITIAL SPARSITY (SAMPLE MASK): " + str(sparsity_level.item()))

                sparsity_level = 1 - (c_shaped_max.count_nonzero() / c_shaped_max.numel())
                self._print_if_verbose("INITIAL SPARSITY (MAX MASK): " + str(sparsity_level.item()))
            else:
                c_shaped = self.c.sample_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = c_shaped.unsqueeze(0).cpu()

                self._save_images(c_out, ["Actual_00"], c_path)

                #NOTE sparsity level is the proportion of zeros in the image
                sparsity_level = 1 - (c_shaped.count_nonzero() / c_shaped.numel())
                self._print_if_verbose("INITIAL SPARSITY: " + str(sparsity_level.item()))

    def test(self):
        """
        Run through the test set.
        We want to save the metrics for each individual sample here!
        We also want to save images of every sample, reconstruction, measurement, recon_meas,
            and the c.
        """
        self._print_if_verbose("TESTING")

        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            _ = self._sample_and_set_mask(bs)            

            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")

            #logging and saving
            self._step_end_logging(i, "test", item, x_hat, x, y)

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test")
        self._add_metrics_to_tb("test")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")

        #grab the raw metrics dictionary and save it
        test_metrics = self.metrics.test_metrics['iter_'+str(self.global_epoch)]
        save_to_pickle(test_metrics, os.path.join(self.log_dir, "test_"+str(self.global_epoch)+".pkl"))

        return

    def run_meta_opt(self):
        finished_flag = False
        
        for iter in tqdm(range(self.hparams.opt.num_iters)):
            #checkpoint
            if iter % self.hparams.opt.checkpoint_iters == 0:
                self._checkpoint()

            #train
            finished_flag = self._run_outer_step()
            self._add_metrics_to_tb("train")
            
            #validate
            if (iter + 1) % self.hparams.opt.val_iters == 0:
                self._run_validation()
                self._add_metrics_to_tb("val")

            self.global_epoch += 1
            
            #check if done
            if finished_flag is True:
                break

        #test
        self._run_test()
        self._add_metrics_to_tb("test")

        self._checkpoint()
    
    def _posterior_loss(self, item, net, iter):
        # Grab the necessary variables and operators
        x = item['gt_image'].to(self.device) #[N, 2, H, W] float, x*
        y = item['ksp'].type(torch.cfloat).to(self.device) #[N, C, H, W] complex, FSx*
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])
        s_maps = item['s_maps'].to(self.device) #[N, C, H, W] complex, S
        
        self.A = MulticoilForwardMRINoMask(s_maps) #FS, [N, 2, H, W] float --> [N, C, H, W] complex
        
        #make noisy sample and get the posterior mean estimate
        x_t, sigma_t = make_noisy_sample(x)
        
        x_hat = single_step_posterior_estimate(net=net, x_t=x_t, sigma_t=sigma_t, FSx=y, P=self.cur_mask_sample,
                                               S=s_maps, likelihood_step_size=self.hparams.net.training_step_size)
        
        #optimisation step
        k = getattr(self.hparams.opt, 'k', 300)
        
        meta_loss = self._get_meta_loss(x_hat, x)
        meta_loss.backward()
        
        self.opt.step() #NOTE temporary for testing both gd and ours
        finished_flag = self.c.max_min_dist_step(k=k)
        # finished_flag = False #comment line above and uncomment next two lines for gumbel-LOUPE
        # self.opt.step() 
        self.opt.zero_grad() 
        
        # Log Things
        with torch.no_grad():
            grad_metrics_dict = {"meta_loss": np.array([meta_loss.item()] * x.shape[0]),
                                 "sigma_t": sigma_t.squeeze().detach().cpu().numpy(),
                                 "k": np.array([k] * x.shape[0]),
                                }
            self.metrics.add_external_metrics(grad_metrics_dict, self.global_epoch, "train")
        
        return x_hat, x, y, finished_flag
    
    def _get_meta_loss(self, x_hat, x):
        """
        Calculates loss between a reconstrution x_hat and true x.
        
        Args:
            x_hat ([N, 2, H, W] real-valued Torch Tensor): The unnormalized predictions.
            x ([N, 2, H, W] real-valued Torch Tensor): The unnormalized ground truth images. 
        
        Returns:
            meta_loss (Torch Tensor): The meta loss - sample-wise mean of individual losses.
                                      Type of meta loss to use is determined by self.hparams.mask.meta_loss_type.
        """
        if self.hparams.mask.meta_loss_type == "l2":
            #Sample-Wise Mean Normalised SSE
            numerator = torch.sum(torch.square(x_hat - x), dim=(1,2,3))
            denominator = torch.sum(torch.square(x), dim=(1,2,3))
            meta_loss = torch.mean(numerator / denominator)
        elif self.hparams.mask.meta_loss_type == "l1":
            #Sample-Wise Mean Normalised SAE
            numerator = torch.sum(torch.abs(x_hat - x), dim=(1,2,3))
            denominator = torch.sum(torch.abs(x), dim=(1,2,3))
            meta_loss = torch.mean(numerator / denominator)
        elif self.hparams.mask.meta_loss_type == "ssim":
            #Sample-wise Mean SSIM
            #Manually iterate instead of using batched solution since data_range argument of 
            #   metric does not accept per-sample pixel ranges   
            pred = torch.norm(x_hat, dim=1, keepdim=True) #[N,1,H,W] Magnitude image
            target = torch.norm(x, dim=1, keepdim=True)
            ssim_loss_list = []
            for i in range(target.shape[0]):
                #The double slice indexing [[i]] slices and keeps dimension intact
                ssim_loss_list.append((1 - structural_similarity_index_measure(preds=pred[[i]], 
                                                                                target=target[[i]], 
                                                                                reduction="sum")))
            meta_loss = torch.mean(torch.stack(ssim_loss_list))
        else:
            raise NotImplementedError("META LOSS NOT IMPLEMENTED!")
        
        return meta_loss

    def _run_outer_step(self):
        self._print_if_verbose("\nTRAINING\n")
        
        finished_flag = False

        for i, (item, x_idx) in tqdm(enumerate(self.train_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            _ = self._sample_and_set_mask(bs)
            
            if self.hparams.net.model in ['dps', 'diffusion_cg']:
                x_hat, x, y, finished_flag = self._posterior_loss(item, self.recon_alg.net, i)
                self._add_batch_metrics(x_hat, x, y, "train")
            else:
                raise NotImplementedError("Model type unsupported")

            #logging and saving
            self._step_end_logging(i, "train", item, x_hat, x, y)
            
            #stop optimization if we have met the desired acceleration
            if finished_flag is True:
                self._print_if_verbose("\n", "FINAL ACCELERATION REACHED. STOPPING OPTIMIZATION", "\n")
                break

        self.metrics.aggregate_iter_metrics(self.global_epoch, "train")
        if (self.global_epoch % self.hparams.opt.checkpoint_iters == 0 or self.global_epoch == self.hparams.opt.num_iters - 1):
            self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "train"), "\n")
        
        if finished_flag is True:
            self._print_if_verbose("\n", "TEARING DOWN TRAINING SET", "\n")
            self.train_loader.dataset.dataset.teardown()
        
        return finished_flag

    def _run_validation(self):
        self._print_if_verbose("\nVALIDATING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.val_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            _ = self._sample_and_set_mask(bs)
            
            #Grab the recon
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "val")

            #logging and saving
            self._step_end_logging(i, "val", item, x_hat, x, y)

        self.metrics.aggregate_iter_metrics(self.global_epoch, "val")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "val"), "\n")
        
        #track the best validation stats
        cur_val_psnr = self.metrics.get_all_metrics(self.global_epoch, "val")['mean_psnr']
        if cur_val_psnr > self.best_val_psnr:
            self.best_val_psnr = cur_val_psnr
            # self.best_val_weights = self.c.weights.clone().detach()
            self.best_val_epoch = self.global_epoch
            self._print_if_verbose("BEST VALIDATION PSNR: ", cur_val_psnr)
        # else:
        #     self._print_if_verbose("Val metrics did not improve; projecting mask weights")
        #     self.c.normalize_probs()

    def _run_test(self):
        self._print_if_verbose("\nTESTING\n")
        
        #Restore best weights
        # self.c.weights.requires_grad_(False)
        # if self.best_val_weights is not None:
        #     self.c.weights.copy_(self.best_val_weights)
        #     self._print_if_verbose("Restoring best validation weights from epoch ", self.best_val_epoch)

        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            _ = self._sample_and_set_mask(bs)
            
            #Grab the recon            
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")

            #logging and saving
            self._step_end_logging(i, "test", item, x_hat, x, y)

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")
    
    def _step_end_logging(self, iter_idx, iter_type, item, x_hat, x, y):
        if iter_type=="test" or iter_idx == 0:
            scan_idxs = item['scan_idx']
            slice_idxs = item['slice_idx']
            x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
        
            manual_save_flag = (iter_type=="test" and iter_idx==0)
            self._save_all_images(x_hat, x, y, x_idx, iter_type, save_masks_manual=manual_save_flag)
        
        return    

    def _shared_step(self, item):
        x = item['gt_image'].to(self.device) #[N, 2, H, W] float second channel is (Re, Im)
        y = item['ksp'].type(torch.cfloat).to(self.device) #[N, C, H, W, 2] float last channel is ""
        s_maps = item['s_maps'].to(self.device) #[N, C, H, W] complex

        #set coil maps and forward operator including current coil maps
        self.recon_alg.H_funcs.s_maps = s_maps
        self.A = MulticoilForwardMRINoMask(s_maps)
        
        steps = 100
        sigma_max = 80.0
        sigma_min = 0.002
        rho = 7.0
        
        net = self.recon_alg.net
        P = self.recon_alg.c.detach()
        
        S_churn=40.
        S_min=0.
        S_max=float('inf')
        S_noise=1.
        
        ##Brains
        # alg_type = "repaint"
        # sigma_max = 1.0
        # config = {}
        
        ##Knees
        # alg_type = "repaint"
        # config = {}
        
        ##Brains and Knees (untested for knees)
        # alg_type = "shallow_dps"
        # config = {'likelihood_step_size': 10.0,
        #           'normalize_grad': True}
        
        # #Brains
        # alg_type = "dps"
        # S_churn=0.
        # config = {'likelihood_step_size': 10.0,
        #           'normalize_grad': True}
        
        #Knees
        alg_type = "dps"
        S_churn=0.
        config = {'likelihood_step_size': 5.0,
                  'normalize_grad': True}
        
        ##Brains and Knees
        # alg_type = "cg"
        # config = {"cg_lambda": 0.3,
        #           "cg_max_iter": 5,
        #           "cg_eps": 0.000001}
        
        t_steps = get_noise_schedule(steps, sigma_max, sigma_min, rho, net, self.device)
        
        x_init = torch.randn_like(x) * t_steps[0]
        
        x_hat = MRI_diffusion_sampling(net=net, x_init=x_init, t_steps=t_steps, FSx=y, P=P, S=s_maps, alg_type=alg_type,
                                       S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise, **config)

        return x_hat, x, y

    @torch.no_grad()
    def _add_batch_metrics(self, x_hat, x, y, iter_type):
        #calc the measurement loss in fully-sampled kspace
        resid = self.A(x_hat) - y
        real_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise SSE

        #calc the measurement loss in the observed indices                                           
        if self.prob_c:
            c_shaped = self.cur_mask_sample.detach().clone() #[N, 1, H, W]
        else:
            c_shaped = self.c.sample_mask().unsqueeze(0).unsqueeze(0) #[1, 1, H, W]
        resid = c_shaped * resid
        weighted_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise SSE with mask

        #calc the ground truth L2 and L1 error
        resid = x_hat - x
        gt_mse = torch.mean(torch.square(resid), dim=[1,2,3]) #element-wise MSE in pixel-space
        gt_mae = torch.mean(torch.abs(resid), dim=[1,2,3]) #element-wise mean MAE 

        extra_metrics_dict = {"real_meas_sse": real_meas_loss.cpu().numpy().flatten(),
                            "weighted_meas_sse": weighted_meas_loss.cpu().numpy().flatten(),
                            "gt_mse": gt_mse.cpu().numpy().flatten(),
                            "gt_mae": gt_mae.cpu().numpy().flatten()}
        
        if self.prob_c:
            prob_mask = self.c.get_prob_mask()
            extra_metrics_dict["mean_prob"] = np.array([torch.mean(prob_mask).item()] * x.shape[0])

            max_mask = self.c.get_max_mask()
            sparsity_level_max = 1 - (max_mask.count_nonzero() / max_mask.numel())
            extra_metrics_dict["sparsity_level_max"] = np.array([sparsity_level_max.item()] * x.shape[0]) #ugly artifact

            cur_mask = self.cur_mask_sample.detach().clone() #[N, 1, H, W]
            sparsity_level_sample = 1 - (cur_mask.count_nonzero(dim=(1,2,3)) / (cur_mask.shape[-1]*cur_mask.shape[-2])) #[N]
            extra_metrics_dict["sparsity_level_sample"] = sparsity_level_sample.cpu().numpy()
        else:
            sparsity_level = 1 - (c_shaped.count_nonzero() / c_shaped.numel())
            extra_metrics_dict["sparsity_level"] = np.array([sparsity_level.item()] * x.shape[0]) 

        self.metrics.add_external_metrics(extra_metrics_dict, self.global_epoch, iter_type)
        self.metrics.calc_iter_metrics(x_hat, x, self.global_epoch, iter_type)

    @torch.no_grad()
    def _save_all_images(self, x_hat, x, y, x_idx, iter_type, save_masks_manual=False):
        if self.hparams.debug or (not self.hparams.save_imgs):
            return
        elif iter_type == "train" and not (self.global_epoch % self.hparams.opt.checkpoint_iters == 0 or
                 self.global_epoch == self.hparams.opt.num_iters - 1):
            return

        #(1) Save samping masks
        if iter_type == "train" or save_masks_manual:
            if self.prob_c:
                c_shaped = self.c.get_prob_mask()
                c_shaped_binary = self.cur_mask_sample.detach().clone()
                c_shaped_max = self.c.get_max_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_max.unsqueeze(0).cpu()])

                self._save_images(c_out, ["Prob_" + str(self.global_epoch), 
                                          "Max_" + str(self.global_epoch)], c_path)
                self._save_images(c_shaped_binary.cpu(), 
                                  ["Sample_"+str(self.global_epoch)+"_"+str(j) for j in range(c_shaped_binary.shape[0])],
                                  c_path)
            else:
                c_shaped = self.c.sample_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = c_shaped.unsqueeze(0).cpu()

                self._save_images(c_out, ["Actual_" + str(self.global_epoch)], c_path)

        #(2) Save reconstructions at every iteration
        meas_recovered_path = os.path.join(self.image_root, iter_type + "_recon_meas", "epoch_"+str(self.global_epoch))
        recovered_path = os.path.join(self.image_root, iter_type + "_recon", "epoch_"+str(self.global_epoch))

        x_hat_vis = torch.norm(x_hat, dim=1).unsqueeze(1) #[N, 1, H, W]
        x_resid = torch.norm(x_hat - x, dim=1).unsqueeze(1) #save the residual image
        x_resid_stretched = (x_resid - torch.amin(x_resid, dim=(1,2,3), keepdim=True)) / \
                                (torch.amax(x_resid, dim=(1,2,3), keepdim=True) - torch.amin(x_resid, dim=(1,2,3), keepdim=True))

        self._save_images(x_hat_vis, x_idx, recovered_path)
        self._save_images(x_resid, [idx + "_resid" for idx in x_idx], recovered_path)
        self._save_images(x_resid_stretched, [idx + "_resid_stretched" for idx in x_idx], recovered_path)
        
        #grab the dict and save the stats for the recons
        metric_dict = self.metrics.get_dict(iter_type)['iter_' + str(self.global_epoch)]
        psnr_array = metric_dict['psnr'][-len(x_idx):]
        ssim_array = metric_dict['ssim'][-len(x_idx):]
        sample_metric_dicts = [{"Slice": idx, "PSNR": psnr_array[i], "SSIM": ssim_array[i]} for i, idx in enumerate(x_idx)]
        metric_path = os.path.join(recovered_path, "sample_metrics.json")
        with open(metric_path, 'a') as f:
            json.dump(sample_metric_dicts, f, indent=4)
            
        avg_metric_dict = [{"MEAN PSNR": np.mean(metric_dict['psnr']), "MEAN SSIM": np.mean(metric_dict['ssim']),
                            "STD PSNR": np.std(metric_dict['psnr']), "STD SSIM": np.std(metric_dict['ssim'])}]
        avg_metric_path = os.path.join(recovered_path, "avg_sample_metrics.json")
        with open(avg_metric_path, 'w') as f:
            json.dump(avg_metric_dict, f, indent=4)

        fake_maps = torch.ones_like(x)[:,0,:,:].unsqueeze(1) #[N, 1, H, W]
        recon_meas = MulticoilForwardMRINoMask(fake_maps)(x_hat)
        recon_meas = torch.abs(recon_meas)

        self._save_images(recon_meas, x_idx, meas_recovered_path)

        #(3) Save ground truth only once
        if "test" in iter_type or self.global_epoch == 0:
            true_path = os.path.join(self.image_root, iter_type)
            meas_path = os.path.join(self.image_root, iter_type + "_meas")

            x_vis = torch.norm(x, dim=1).unsqueeze(1) #[N, 1, H, W]

            self._save_images(x_vis, x_idx, true_path)

            gt_meas = MulticoilForwardMRINoMask(fake_maps)(x)
            gt_meas = torch.abs(gt_meas)

            self._save_images(gt_meas, x_idx, meas_path)

    def _init_dataset(self):
        train_set, test_set = get_dataset(self.hparams)
        split_dict = split_dataset(train_set, test_set, self.hparams)
        train_dataset = split_dict['train']
        val_dataset = split_dict['val']
        test_dataset = split_dict['test']

        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.data.train_batch_size, shuffle=True,
                                num_workers=16, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=16, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.hparams.data.test_batch_size, shuffle=False,
                                num_workers=16, drop_last=True)

    def _init_c(self):
        num_acs_lines = getattr(self.hparams.mask, 'num_acs_lines', 20)

        if self.prob_c:
            self.c = Probabilistic_Mask(self.hparams, self.device, num_acs_lines)
        else:
            self.c = Baseline_Mask(self.hparams, self.device, num_acs_lines)

        return
    
    def _sample_and_set_mask(self, num_samples, set_recon_alg=True):
        """
        Grabs num_samples sampling patterns, sets them as current, updates the recon_alg patterns,
            and returns a stopgrad version of the patterns.
            
        Args:
            num_samples (int): Number of patterns to sample.
            set_recon_alg (bool, default=True): Whether to update the recon algorithm with the new patterns.
        
        Returns:
            c_shaped ([N, 1, H, W] real-valued Torch Tensor): The (detached) patterns that were sampled.  
        """
        if self.prob_c:
            tau = getattr(self.hparams.mask, 'tau', 0.5)
            self.cur_mask_sample = torch.stack([self.c.sample_mask(tau).unsqueeze(0) for _ in range(num_samples)], dim=0)
            c_shaped = self.cur_mask_sample.detach().clone()
        else:
            c_shaped = self.c.sample_mask().repeat(num_samples, 1, 1, 1)
        
        if set_recon_alg:
            self.recon_alg.set_c(c_shaped)
        
        return c_shaped

    def _init_meta_optimizer(self):
        opt_type = self.hparams.opt.optimizer
        lr = self.hparams.opt.lr

        if opt_type == 'adam':
            meta_opt = torch.optim.Adam([{'params': self.c.weights}], lr=lr)
        elif opt_type == 'sgd':
            meta_opt = torch.optim.SGD([{'params': self.c.weights}], lr=lr)
        else:
            raise NotImplementedError("Optimizer not supported!")

        if self.hparams.opt.decay:
            meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_opt, self.hparams.opt.lr_decay)
        else:
            meta_scheduler = None

        self.opt =  meta_opt
        self.scheduler = meta_scheduler

    def _checkpoint(self):
        if self.hparams.debug:
            return

        save_dict = {
            "c_weights": self.c.weights.detach().cpu(),
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
        self._print_if_verbose("RESUMING FROM CHECKPOINT")

        checkpoint = load_if_pickled(os.path.join(self.log_dir, "checkpoint.pkl"))
        metrics = load_if_pickled(os.path.join(self.log_dir, "metrics.pkl"))

        self.c.weights.copy_(checkpoint["c_weights"].to(self.device))
        self.c.weights.requires_grad_()

        if not(self.args.baseline) and not(self.args.test):
            self.opt.load_state_dict(checkpoint['opt_state'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.global_epoch = checkpoint['global_epoch']

        self.metrics.resume(metrics)

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
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_images(images, img_indices, save_path)

    def _print_if_verbose(self, *text):
        if self.hparams.verbose:
            print("".join(str(t) for t in text))
