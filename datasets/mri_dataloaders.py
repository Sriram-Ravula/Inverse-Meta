from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import h5py
import sigpy as sp
import pickle as pkl
import sys

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=-3) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=-3))

class BrainMultiCoil(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 image_size=384,
                 num_slices=None,
                 slice_mapper=None,
                 load_slice_info=False,
                 save_slice_info=False,
                 kspace_pad=28):
        # Attributes
        self.file_list    = file_list
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.image_size = image_size

        self.kspace_pad = kspace_pad

        if not load_slice_info:
            # Comment next two blocks if we only want central 5 slices
            # Access meta-data of each scan to get number of slices
            print("Reading " + str(len(self.file_list)) + " Scans")
            self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
            for idx, file in tqdm(enumerate(self.file_list)):
                input_file = os.path.join(self.input_dir, os.path.basename(file))
                with h5py.File(input_file, 'r') as data:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

            # Create cumulative index for mapping
            self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'
        else:
            print("Loading available scan information\n")
            self.num_slices = np.load(num_slices)
            self.slice_mapper = np.load(slice_mapper)
        
        if save_slice_info:
            print("Saving compiled scan information!\n")
            np.save(num_slices, self.num_slices)
            np.save(slice_mapper, self.slice_mapper)

    def __len__(self):
        #Comment this block if we only want central 5 slices
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

        # #Uncomment this return block if we only want the central five slices
        # return int(len(self.file_list) * 5)

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Comment this block if we want only central 5 slices
        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # # Uncomment this block if we want only central 5 slices
        # # Get scan and slice index
        # # we only use the 5 central slices
        # scan_idx = idx // 5 
        # # Offset from cumulative range
        # slice_idx = idx % 5

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            # Get maps

            # Uncomment block if only want central 5 slices
            # num_slices = int(data['s_maps'].shape[0])
            # slice_idx_shifted = (num_slices // 2) - 2 + slice_idx
            # s_maps = np.asarray(data['s_maps'][slice_idx_shifted])

            # Comment block if only want central 5 slices
            s_maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            # Get k-space

            # Uncomment block if only want central 5 slices
            # num_slices = int(data['kspace'].shape[0])
            # slice_idx_shifted = (num_slices // 2) - 2 + slice_idx
            # gt_ksp = np.asarray(data['kspace'][slice_idx_shifted])

            # Comment block if only want central 5 slices
            gt_ksp = np.asarray(data['kspace'][slice_idx])

        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], gt_ksp.shape[1], self.image_size))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size,
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV in phase-encode
        s_maps = sp.fft(s_maps, axes=(-2, -1)) # These are now maps in k-space
        s_maps = sp.resize(s_maps, (
            s_maps.shape[0], s_maps.shape[1], self.image_size))

        # Reduce FoV by half in the readout direction
        s_maps = sp.ifft(s_maps, axes=(-2,))
        s_maps = sp.resize(s_maps, (s_maps.shape[0], self.image_size,
                                    s_maps.shape[2]))
        s_maps = sp.fft(s_maps, axes=(-2,)) # Back to k-space
        s_maps = sp.ifft(s_maps, axes=(-2, -1)) # Finally convert back to image domain

        # find mvue image
        gt_mvue = get_mvue(gt_ksp, s_maps)

        ksp = gt_ksp
        # find mvue image
        aliased_mvue = get_mvue(ksp, s_maps)

        # scale_factor = np.percentile(np.abs(aliased_mvue), 99)
        gt_mvue_scale_factor = np.percentile(np.abs(gt_mvue),99)
        ksp /= gt_mvue_scale_factor
        aliased_mvue /= gt_mvue_scale_factor
        gt_mvue /= gt_mvue_scale_factor

        # s_maps_scale = np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=-3))
        # # print(s_maps_scale)
        # ksp /= s_maps_scale
        # aliased_mvue /= s_maps_scale
        # gt_mvue /= s_maps_scale
        # s_maps /= s_maps_scale

        # Apply ACS-based instance scaling
        aliased_mvue_two_channel = np.float16(np.zeros((2,) + aliased_mvue.shape))
        aliased_mvue_two_channel[0] = np.float16(np.real(aliased_mvue))
        aliased_mvue_two_channel[1] = np.float16(np.imag(aliased_mvue))

        gt_mvue_two_channel = np.float16(np.zeros((2,) + gt_mvue.shape))
        gt_mvue_two_channel[0] = np.float16(np.real(gt_mvue))
        gt_mvue_two_channel[1] = np.float16(np.imag(gt_mvue))

        #Apply optional K-space padding
        #This allows us to make the non-batch dimensions of all the samples homogenous,
        #   and allows for batch size > 1
        if self.kspace_pad:
            if (ksp.shape[0] < self.kspace_pad) and (s_maps.shape[0] < self.kspace_pad):
                ksp = np.pad(ksp, ((0,self.kspace_pad - ksp.shape[0]), (0,0), (0,0)))
                s_maps = np.pad(s_maps, ((0,self.kspace_pad - s_maps.shape[0]), (0,0), (0,0)))

        # Output
        sample = {
                  'ksp': ksp, #[C, H, W] complex64 numpy array
                  's_maps': s_maps, #[C, H, W] complex64 numpy array
                  # 'mask': mask,
                  'aliased_image': aliased_mvue_two_channel.astype(np.float32),
                  'gt_image': gt_mvue_two_channel.astype(np.float32),
                  'scale_factor': gt_mvue_scale_factor.astype(np.float32),
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}

        return sample, idx

class KneesMultiCoil(BrainMultiCoil):
    def __init__(self, file_list, maps_dir, input_dir,
                 image_size=320,
                 num_slices=None,
                 slice_mapper=None,
                 load_slice_info=False,
                 save_slice_info=False,
                 kspace_pad=False):
        super(KneesMultiCoil, self).__init__(file_list, maps_dir, input_dir,
                                             image_size, num_slices, slice_mapper, 
                                             load_slice_info, save_slice_info, kspace_pad)
