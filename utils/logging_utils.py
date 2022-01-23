"""
Class used to help with tensorboard logging, checkpointing and loading, and statistics saving.
"""
import torch.utils.tensorboard as tb
import torch
import numpy as np
import sys
import os
import yaml
from PIL import Image
import torch.nn.functional as F
import torchvision
import pickle

from utils.metrics_utils import Metrics
from learners.meta_learner import MetaLearner
from loss_utils import get_inpaint_mask

def save_image(image, path):
    """Save a pytorch image as a png file"""
    image = image.detach().cpu().numpy() #image comes in as an [C, H, W] torch tensor
    x_png = np.uint8(np.clip(image*256,0,255))
    x_png = x_png.transpose(1,2,0)
    if x_png.shape[-1] == 1:
        x_png = x_png[:,:,0]
    x_png = Image.fromarray(x_png).save(path)

def save_images(est_images, save_prefix):
    """Save a batch of images (in a dictionary) to png files"""
    for image_num, image in est_images.items():
        save_image(image, os.path.join(save_prefix, str(image_num), '.png'))

def save_measurement_images(est_images, hparams, save_prefix):
    """Save a batch of image measurements to png files"""
    A_type = hparams.problem.measurement_type

    if A_type not in ['superres', 'inpaint']:
        print("Can't save given measurement type")
        return

    for image_num, image in est_images.items():
        if A_type == 'superres':
            image = image * get_inpaint_mask(hparams)
        elif A_type == 'inpaint':
            image = F.avg_pool2d(image, hparams.problem.downsample_factor)
            image = F.interpolate(image, scale_factor=hparams.problem.downsample_factor)
        save_image(image, os.path.join(save_prefix, str(image_num), '.png'))

def save_to_pickle(data, pkl_filepath):
    """Save the data to a pickle file"""
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data

class Logger:
    """
    Class for saving, checkpointing, and logging various things
    """
    def __init__(self, metrics: Metrics, learner: MetaLearner, hparams, log_dir):
        self.log_dir = log_dir
        self.hparams = hparams
        self.image_root = os.path.join(log_dir, 'images')
        self.tb_root = os.path.join(log_dir, 'tensorboard')
        self.metrics_root = os.path.join(log_dir, 'metrics')

        self.metrics = metrics
        self.learner = learner

        self.__make_log_folder()
        self.__save_config()

        self.tb_logger = tb.SummaryWriter(log_dir=log_dir)

    def __make_log_folder(self):
        if os.path.exists(self.log_dir):
            print("Folder exists. Program halted.")
            sys.exit(0)
        else:
            os.makedirs(self.log_dir)
            os.makedirs(self.image_root)
            os.makedirs(self.tb_root)
            os.makedirs(self.metrics_root)

    def __save_config(self):
        with open(os.path.join(self.log_dir, 'config.yml'), 'w') as f:
            yaml.dump(self.hparams, f, default_flow_style=False)
    
    def checkpoint(self):
        checkpoint_dict = self.get_checkpoint_dict()
        metrics_dict = self.get_metrics_dict()
        if hasattr(self.learner, 'meta_scheduler'):
            states = [
                self.learner.meta_opt.state_dict(),
                self.learner.meta_scheduler.state_dict()
            ]
        else:
            states = [
                self.learner.meta_opt.state_dict()
            ]

        save_to_pickle(metrics_dict, os.path.join(self.metrics_root, 'metrics'))
        save_to_pickle(checkpoint_dict, os.path.join(self.log_dir, 'checkpoint'))
        torch.save(states, os.path.join(self.log_dir, states, '.pth'))

        return
    
    def get_checkpoint_dict(self):
        out_dict = {
            'A': self.learner.A,
            'global_iter': self.learner.global_iter,
            'best_iter': self.learner.best_iter,
            'c_list': self.learner.c_list,
            'c': self.learner.c,
            'grad_norms': self.learner.grad_norms,
            'grads': self.learner.grads
        }
        return out_dict
    
    def get_metrics_dict(self):
        out_dict = {
            'range': self.metrics.range,
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

        return out_dict
    
    def save_image_measurements(self, images, image_nums, save_prefix):
        save_path = os.path.join(self.image_root, save_prefix)

        image_dict = {}

        for i in range(images.shape[0]):
            image_dict[image_nums[i]] = images[i]
        
        save_measurement_images(image_dict, self.hparams, save_path)

    def save_images(self, images, image_nums, save_prefix):
        save_path = os.path.join(self.image_root, save_prefix)

        image_dict = {}

        for i in range(images.shape[0]):
            image_dict[image_nums[i]] = images[i]
        
        save_images(image_dict, save_path)
    
    def save_image_measurements_torch(self, images, image_nums, save_prefix):
        A_type = self.hparams.problem.measurement_type
        save_path = os.path.join(self.image_root, save_prefix + '.pth')

        if A_type not in ['superres', 'inpaint']:
            print("Can't save given measurement type")
            return

        if A_type == 'superres':
            images = images * get_inpaint_mask(self.hparams)
        elif A_type == 'inpaint':
            images = F.avg_pool2d(images, self.hparams.problem.downsample_factor)
            images = F.interpolate(images, scale_factor=self.hparams.problem.downsample_factor)
        
        torch.save(images, save_path)
    
    def save_images_torch(self, images, image_nums, save_prefix):
        save_path = os.path.join(self.image_root, save_prefix + '.pth')
        torch.save(images, save_path)
    
    def add_tb_images(self, images, tag):
        step = self.learner.global_iter
        self.tb_logger(tag, images, global_step=step)

    def add_metrics_to_tb(self, iter_type='train'):
        """
        Run through this Logger's metrics object and log everything there.
        For each type of metric, we want to log the train, val, and test metrics on the same plot.
        Intended to be called at the end of each type of iteration (train, test, val)
        """
        assert iter_type in ['train', 'val', 'test']

        raw_dict = self.metrics.__retrieve_dict(iter_type, dict_type='raw')
        agg_dict = self.metrics.__retrieve_dict(iter_type, dict_type='aggregate')
        best_dict = self.metrics.__retrieve_dict(iter_type, dict_type='best')

        step = self.learner.global_iter
        iterkey ='iter_' + str(step)

        if iterkey not in raw_dict:
            print("current iteration has not yet been logged")
            return
        
        for metric_type, metric_value in raw_dict[iterkey].items():
            for i, val in enumerate(metric_value):
                self.tb_logger.add_scalars("raw " + metric_type, {iter_type: val}, i)
        
        for metric_type, metric_value in agg_dict[iterkey].items():
            self.tb_logger.add_scalars(metric_type, {iter_type: metric_value}, step)
        
        for metric_type, metric_value in best_dict.items():
            self.tb_logger.add_scalars("best " + metric_type + " iter", {iter_type: metric_value[0]}, step)
            self.tb_logger.add_scalars("best " + metric_type + " value", {iter_type: metric_value[1]}, step)
        
        return
