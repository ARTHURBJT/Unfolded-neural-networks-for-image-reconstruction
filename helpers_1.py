import os

import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from PIL import Image
import imageio

import matplotlib.pyplot as plt

import pylab
# from skimage.measure.simple_metrics import compare_psnr

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def create_dir(dir):
    try: # Create target Directory
        os.makedirs(dir)
    except:
        print('folder '+dir+' exists')

def save_image_numpy(im, path):
    """
    Saves an image im np format
    """
    #print(im.shape)
    im_bounded = im*255.
    im_bounded[im_bounded>255.] = 255.
    im_bounded[im_bounded<0.] = 0.
    imageio.imwrite(path, np.uint8(im_bounded))

def save_image(im,path,nrows=10):
    """
    Saves an image im torch format
    """
    with torch.no_grad():
        Img = make_grid(im, nrow=nrows, normalize=True, scale_each=True)
        try:
            Img = Img.cpu().numpy()
        except:
            Img = Img.detach().cpu().numpy()
        Img = np.moveaxis(Img, 0, -1)
        imageio.imwrite(path, np.uint8(Img*255.))


def save_several_images(ims,paths,nrows=10,out_format='.png'):
    """
    Saves an image im torch format
    """
    for i,im in enumerate(ims):
        save_image(im, paths[i]+out_format, nrows=nrows)


def create_noise(shape,lev=0.1,fixed=True):
    if fixed==True:
        noise = lev*torch.randn(shape).type(Tensor)
    else:
        noise = torch.zeros(shape).type(Tensor)
        stdN = np.random.uniform(0, 1, size=noise.shape[0])
        for _ in range(noise.shape[0]):
            noise[_,...] = lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor)
    return noise

def snr(x, y):
    """
    snr - signal to noise ratio

       v = snr(x,y);

     v = 20*log10( norm(x(:)) / norm(x(:)-y(:)) )

       x is the original clean signal (reference).
       y is the denoised signal.

    Copyright (c) 2014 Gabriel Peyre
    """

    return 20 * np.log10(pylab.norm(x) / pylab.norm(x - y))

def snr_numpy(xtrue, x):
    snr = 20* np.log10(np.linalg.norm(xtrue.flatten())/np.linalg.norm(xtrue.flatten()-x.flatten()))
    return snr

def snr_torch(x, y, compute_std=False):
    '''
    snr - signal to noise ratio

       v = snr(x,y);

     v = 20*log10( norm(x(:)) / norm(x(:)-y(:)) )

       x is the original clean signal (reference).
       y is the denoised signal.
    '''
    s, s_square = 0, 0
    c = 0
    for _ in range(x.shape[0]):
        x_np = x[_,0,...].detach().cpu().numpy()
        y_np = y[_,0,...].detach().cpu().numpy()
        s+=snr(x_np,y_np)
        if compute_std:
            s_square+=s**2
        c+=1
    s_tot = s/float(c)
    if compute_std:
        var = (s_square-s**2)/float(c)
        return s_tot, np.sqrt(var)
    else:
        return s_tot


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', load_model_only=False):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    # From https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
    if load_model_only or optimizer==None:
        checkpoint = torch.load(filename,map_location=lambda storage, loc: storage)
        model.module.load_state_dict(checkpoint.module.state_dict())
        return model
    else:
        start_epoch = 0
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model, optimizer, start_epoch

def get_image(path='test_pictures/lena.png', even=True, n_ch=3):
    im_np = imageio.imread(path)
    im_np = np.asarray(im_np[:,:,0:n_ch], dtype="float32")
    im_np = np.moveaxis(im_np,-1,0)
    im_np /= im_np.max()
    sample = torch.from_numpy(im_np).type(Tensor)

    sample.unsqueeze_(0)

    x_bd = 2*(sample.shape[-2]//2) if even else -1
    y_bd = 2*(sample.shape[-1]//2) if even else -1

    return sample[...,:x_bd, :y_bd]


def imshow(img, title=None):
    #npimg = img.copy()
    npimg = img.detach().cpu().numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()