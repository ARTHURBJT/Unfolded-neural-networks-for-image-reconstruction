
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
from torch.utils.data.sampler import SubsetRandomSampler

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
    def __init__(self, path_data, transform=None, reduction_size=None, pattern='*/*.JPEG', channels=3):
        self.transform = transform
        self.channels = channels
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

        if len(image_true.shape)<3:
            image_true = np.repeat(image_true[:, :, np.newaxis], 3, axis=2)
        else:
            if image_true.shape[2]>3: # Strange case that seems to appear at least once
                image_true = image_true[:, :, :3]

        if self.transform is not None:
            image_true = Image.fromarray(np.uint8(image_true))
            image_true = self.transform(image_true)
            image_true = np.asarray(image_true, dtype="float32")

        if self.channels == 1: # RGB conversion
            image_true = 0.2125*image_true[..., 0:1]+0.7154*image_true[..., 1:2]+0.0721*image_true[..., 2:3]

        image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0))
 
        return {'image_true': image_true, 'image_name': image_name}

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_dataset(path_dataset='../../data/imagenet_32/', path_red_dataset=None, random_seed=30, bs=50, patchSize=64, red=False, red_size=25, train_mode=True, channels=3, pattern_red='*/*.JPEG', num_workers=6):
    '''
    Returns the dataloader for the dataset
    '''

    data_transform_true = transforms.Compose([
        transforms.RandomResizedCrop(patchSize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    #img_dataset = imnetDataset(path_data=path_dataset, transform=data_transform_true, reduction_size=None, channels=channels, pattern=pattern_red)
    img_dataset = customMNIST(
        root ="./data",
        train = True,
        transform = transforms.ToTensor(),
        download = True
    )



    validation_split = .02
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(img_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    torch.manual_seed(random_seed)

    # Creating samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader_train = torch.utils.data.DataLoader(img_dataset, batch_size=bs, 
                                            sampler=train_sampler, num_workers=num_workers)
    loader_val = torch.utils.data.DataLoader(img_dataset, batch_size=bs,
                                                    sampler=valid_sampler, num_workers=num_workers)
    N = int(img_dataset[0]['image_true'].shape[0])
    return loader_train, loader_val, N

#     if red and path_red_dataset is not None:
#         if train_mode:
#             transf_red = transforms.Compose([
#             transforms.CenterCrop(256)
#             ])
#             bs = 25
#         else:
#             transf_red = None
#             bs = 1
#         img_dataset_reduced = imnetDataset(path_data=path_red_dataset, transform=transf_red, reduction_size=red_size, channels=channels, pattern=pattern_red)
#         loader_red = torch.utils.data.DataLoader(img_dataset_reduced, batch_size=1)
#         return loader_train, loader_val, loader_red
#     else:
#         return loader_train, loader_val




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
        ).data[0]

    def __getitem__(self, index):
        """ 
        Get a sample from the dataset
        """
        image_true = (self.data / 255.).flatten()
        #image_true = (self.data[index] / 255.).flatten()
        #image_true = np.asarray(image_true, dtype="float32")
        image_name = index

        #image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0)).flatten()
        return {'image_true': image_true, 'image_name': image_name}
    

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return 8192
        return len(self.data)









def get_test_dataloader(path_dataset='../datasets/BSDS300/', batch_size=1, red_size=None, channels=3, pattern_red='.jpg', num_workers=6):
    """
    Yields the most basic image dataloader
    """
    transf = None
    img_dataset = imnetDataset(path_data=path_dataset, transform=transf, reduction_size=red_size, channels=channels, pattern=pattern_red)
    loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, num_workers=num_workers)
    return loader


class customDataset(D.Dataset):
    """Loads a custom image dataset from 1 to 2 folder(s)
    If only 1 folder is precised, then we consider that the input image is the groundtruth
    If 2 datasets are precised, then we consider that it corresponds to pairs groundtruths/blurred
    """
    def __init__(self, path_data, name, path_blurred=None, transform=None, reduction_size=None, pattern='*.jpg', channels=3):
        self.dataset_name = name
        self.transform = transform
        self.channels = channels

        if reduction_size==None:
            self.filenames = glob.glob(osp.join(path_data, pattern))
            if path_blurred is not None:
                self.filenames_blurred = glob.glob(osp.join(path_blurred, pattern))
        else:
            filenames_list = sorted(glob.glob(osp.join(path_data, pattern)))   
            self.filenames = filenames_list[:reduction_size]
            if path_blurred is not None:
                filenames_blurred_list = sorted(glob.glob(osp.join(path_blurred, pattern)))   
                self.filenames_blurred = filenames_blurred_list[:reduction_size]

        if path_blurred is None:
            self.filenames_blurred = None

        self.len = len(self.filenames)
        
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

        if self.channels == 1: # RGB conversion to grayscale
            image_true = 0.2125*image_true[...,0:1]+0.7154*image_true[...,1:2]+0.0721*image_true[...,2:3]

        image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0))

        if self.filenames_blurred == None:
            return {'image_true': image_true, 'image_name': image_name}
        else:
            image_blurred = Image.open(self.filenames_blurred[index])
            image_blurred = np.asarray(image_blurred, dtype="float32")

            if len(image_true.shape)<3:
                image_blurred = np.repeat(image_blurred[:, :, np.newaxis], 3, axis=2)
            else:
                if image_true.shape[2]>3: # Strange case that seems to appear at least once
                    image_blurred = image_blurred[:, :, :3]

            if self.transform is not None:
                image_blurred = Image.fromarray(np.uint8(image_blurred))
                image_blurred = self.transform(image_blurred)
                image_blurred = np.asarray(image_blurred, dtype="float32")

            if self.channels == 1: # RGB conversion to grayscale
                image_blurred = 0.2125*image_blurred[...,0:1]+0.7154*image_blurred[...,1:2]+0.0721*image_blurred[...,2:3]

            image_blurred = torch.from_numpy(np.moveaxis((image_blurred/255.), -1, 0))

            return {'image_true': image_true, 'image_blurred': image_blurred, 'image_name': image_name}

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len





class noiseDataset(D.Dataset):
    """Loads a custom image dataset from 1 folder
    Adds noise to images
    I am doing this with the view of loading images from a noisy dataset in the end (otherwise could do on the fly)
    """
    def __init__(self, path_data, sigma=0.01, path_blurred=None, transform=None, reduction_size=None, pattern='*.jpg', channels=3):
        self.transform = transform
        self.channels = channels

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

        if self.channels == 1: # RGB conversion to grayscale
            image_true = 0.2125*image_true[...,0:1]+0.7154*image_true[...,1:2]+0.0721*image_true[...,2:3]

        image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0))
        image_noisy = image_true+self.sigma*torch.randn_like(image_true)

        return {'image_true': image_true, 'image_noisy': image_noisy, 'image_name': image_name}
        

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

