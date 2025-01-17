
import glob
import os
import os.path as osp

import numpy as np
import imageio
from PIL import Image
from PIL import ImageFilter

import torch
import torch.nn as nn
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import torchvision
import torchvision.transforms as transforms

# from skimage.measure.simple_metrics import compare_psnr

# from torchvision.utils import make_grid
# import torchvision.transforms.functional as F

# import pylab
# import scipy
# import scipy.io

# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Class for dataset imageNet
# On the validation set

class imnetDataset(D.Dataset):
    """Loads the ImageNet dataset"""
    def __init__(self, path_data, transform=None, reduction_size=None, pattern='*/*.JPEG'):
        self.transform = transform
        if reduction_size==None:
            self.filenames = glob.glob(osp.join(path_data, pattern)) # /gpfsdswork/dataset/imagenet/RawImages/val/
        else:
            filenames_list = sorted(glob.glob(osp.join(path_data, pattern)))   
            self.filenames = filenames_list[:reduction_size]
        self.len = len(self.filenames)
        
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ 
        Get a sample from the dataset
        """
        image_name = os.path.splitext(os.path.basename(self.filenames[index]))[0]
        image_true = Image.open(self.filenames[index])
        image_true = np.asarray(image_true, dtype="float32")

        if self.transform is not None:
            image_true = Image.fromarray(np.uint8(image_true))
            image_true = self.transform(image_true)
            image_true = np.asarray(image_true, dtype="float32")


        image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0))
 
        return {'image_true': image_true, 'image_name': image_name}

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_dataset(random_seed=30, bs=50, train_mode=True, num_workers=6):
    '''
    Returns the dataloader for the dataset
    '''
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    img_dataset = customMNIST(
        root ="./data",
        train = True,
        transform = transforms.ToTensor(),
        download = True
    )

    validation_split = .05
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(img_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    train_sampler = SequentialSampler(train_indices)
    valid_sampler = SequentialSampler(val_indices)

    g = torch.Generator()
    g.manual_seed(random_seed)

    loader_train = torch.utils.data.DataLoader(img_dataset, batch_size=bs, 
                                            sampler=train_sampler, num_workers=num_workers, generator=g)
    loader_val = torch.utils.data.DataLoader(img_dataset, batch_size=bs,
                                                    sampler=valid_sampler, num_workers=num_workers, generator=g)
    N = int(img_dataset[0]['image_true'].shape[0])
    return loader_train, loader_val, N


class customMNIST(D.Dataset):
    """Loads a custom image dataset from 1 to 2 folder(s)
    If only 1 folder is precised, then we consider that the input image is the groundtruth
    If 2 datasets are precised, then we consider that it corresponds to pairs groundtruths/blurred
    """

    def __init__(self, root, train, download, transform=transforms.ToTensor()):
        self.data = torchvision.datasets.MNIST(
            root =root,
            train = train,
            transform = transform,
            download = download
        ).data

    def __getitem__(self, index):
        """ 
        Get a sample from the dataset
        """
        image_true = (self.data[index] / 255.).flatten()
        #image_true = (self.data[index] / 255.).flatten()
        #image_true = np.asarray(image_true, dtype="float32")
        image_name = index

        #image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0)).flatten()
        return {'image_true': image_true, 'image_name': image_name}
    

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.data)





class noiseDataset(D.Dataset):
    """Loads a custom image dataset from 1 folder
    Adds noise to images
    I am doing this with the view of loading images from a noisy dataset in the end (otherwise could do on the fly)
    """
    def __init__(self, path_data, sigma=0.01, transform=None, reduction_size=None, pattern='*.jpg'):
        self.transform = transform

        if reduction_size==None:
            self.filenames = glob.glob(osp.join(path_data, pattern))
        else:
            filenames_list = sorted(glob.glob(osp.join(path_data, pattern)))   
            self.filenames = filenames_list[:reduction_size]

        self.len = len(self.filenames)
        self.sigma = sigma
        
    def __getitem__(self, index):
        """ 
        Get a sample from the dataset
        """
        image_true = Image.open(self.filenames[index])
        image_true = np.asarray(image_true, dtype="float32")
        image_name = os.path.basename(self.filenames[index])

        if len(image_true.shape)<3:
            image_true = np.repeat(image_true[:, :, np.newaxis], 3, axis=2)
        else:
            if image_true.shape[2]>3: # Strange case that seems to appear at least once
                image_true = image_true[:, :, :3]

        if self.transform is not None:
            image_true = Image.fromarray(np.uint8(image_true))
            image_true = self.transform(image_true)
            image_true = np.asarray(image_true, dtype="float32")

        image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0))
        image_noisy = image_true+self.sigma*torch.randn_like(image_true)

        return {'image_true': image_true, 'image_noisy': image_noisy, 'image_name': image_name}
        

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
