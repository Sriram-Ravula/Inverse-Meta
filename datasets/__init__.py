import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import glob

from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.mri_dataloaders import BrainMultiCoil, KneesSingleCoil, KneesMultiCoil

def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

def get_dataset(config):
    if config.data.dataset == 'celeba':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(config.data.data_path, 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()]),
                                 download=True)
        else:
            dataset = CelebA(root=os.path.join(config.data.data_path, 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor()]),
                                 download=True)

        test_dataset = CelebA(root=os.path.join(config.data.data_path, 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor()]),
                                  download=True)

    elif config.data.dataset == "ffhq":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(config.data.data_path),
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()]),
                                resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(config.data.data_path),
                            transform=transforms.ToTensor(),
                            resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == 'Brain-Multicoil':
        files = get_all_files(config.data.input_dir, pattern='*.h5')
        dataset = None
        test_dataset = BrainMultiCoil(files,
                                input_dir=config.data.input_dir,
                                maps_dir=config.data.maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)
        # num_items = len(dataset)
        # indices = list(range(num_items))
        # random_state = np.random.get_state()
        # np.random.seed(2019)
        # np.random.shuffle(indices)
        # np.random.set_state(random_state)
        # train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        # test_dataset = Subset(dataset, test_indices)
        # dataset = Subset(dataset, train_indices)

    elif config.data.dataset == 'Knee-Multicoil':
        files = get_all_files(config.data.input_dir, pattern='*.h5')
        dataset = None
        test_dataset = KneesMultiCoil(files,
                                input_dir=config.data.input_dir,
                                maps_dir=config.data.maps_dir,
                                image_size = config.data.image_size,
                                R=config.problem.R,
                                pattern=config.problem.pattern,
                                orientation=config.problem.orientation)
        # num_items = len(dataset)
        # indices = list(range(num_items))
        # random_state = np.random.get_state()
        # np.random.seed(2019)
        # np.random.shuffle(indices)
        # np.random.set_state(random_state)
        # train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        # test_dataset = Subset(dataset, test_indices)
        # dataset = Subset(dataset, train_indices)
    elif config.data.dataset == 'Knees-Singlecoil':
        dataset = None
        files = get_all_files(config.data.input_dir, pattern='*.h5')
        test_dataset = KneesSingleCoil(files,
                                image_size = config.data.image_size,
                                R=config.data.R,
                                pattern=config.data.pattern,
                                orientation=config.data.orientation)

    else:
        raise NotImplementedError("Dataset not supported!")

    return dataset, test_dataset

def split_dataset(base_dataset, hparams):
    """
    Split a given dataset into train, val, and test sets.
    If we do not want a validation set, returns None for val.

    Args:
        base_dataset: The dataset to use for splitting.
                      Type: Dataset.
        hparams: The experiment parameters to use for splitting.
                 Type: Namespace.
                 Expected to have the constituents:
                    hparams.data.num_train - int
                    hparams.data.num_val - int
                    hparams.data.num_test - int
                    hparams.seed - int

    Returns:
        datasets: A dict containing the train, val, and test datasets.
                  Type: dict.
    """
    num_train = hparams.data.num_train
    num_val = hparams.data.num_val
    num_test = hparams.data.num_test

    indices = list(range(len(base_dataset)))

    # random_state = np.random.get_state()
    # np.random.seed(hparams.seed)
    # np.random.shuffle(indices)
    # np.random.set_state(random_state)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:num_train+num_val+num_test]
    # TODO: delete
    print('INDICES', train_indices, val_indices, test_indices)

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(base_dataset, test_indices)

    out_dict = {'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset}

    return out_dict
