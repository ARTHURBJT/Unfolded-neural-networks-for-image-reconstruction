# Training of an unfolded dual-FB network for Gaussian denoising

import argparse
import os
import pathlib


import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from im_class_conv import get_dataset
from helpers import create_dir, save_several_images, snr_torch, get_image
from modelhelper_conv import get_model, load_model
from sparcity_op import sparcity_op


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()




def inv_im(im):
    return np.ones(im.shape) - im

def imshowgray(im_true,im_noise,im_out, i, title=None):

    def save_high_quality_image(image, filename):
        plt.figure(figsize=(20, 20))  # Taille de la figure (20 x 20 pouces)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')  # Désactiver les axes
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)  # Sauvegarder avec haute qualité
        plt.close()

    if len(im_true.shape) > 2:
        im1 = np.concatenate((np.concatenate((inv_im(im_true[0]),inv_im(im_noise[0])), axis=1),inv_im(im_out[0])), axis=1)
        im2 = np.concatenate((np.concatenate((inv_im(im_true[1]),inv_im(im_noise[1])), axis=1),inv_im(im_out[1])), axis=1)
        im3 = np.concatenate((np.concatenate((inv_im(im_true[2]),inv_im(im_noise[2])), axis=1),inv_im(im_out[2])), axis=1)

        # Ajouter une dimension pour les canaux pour chaque image
        im1 = np.expand_dims(im1, axis=-1)
        im2 = np.expand_dims(im2, axis=-1)
        im3 = np.expand_dims(im3, axis=-1)

        # Sauvegarder les images avec haute qualité
        save_high_quality_image(im1, title + str(i) + '_1' + '.png')
        save_high_quality_image(im2, title + str(i) + '_2' + '.png')
        save_high_quality_image(im3, title + str(i) + '_3' + '.png')


        writer.add_image('Fashion-MNIST Images 1', im1, dataformats='HWC')
        writer.add_image('Fashion-MNIST Images 2', im2, dataformats='HWC')
        writer.add_image('Fashion-MNIST Images 3', im3, dataformats='HWC')
        writer.flush()
        plt.close('all')
    else:
        im = np.concatenate((np.concatenate((inv_im(im_true),inv_im(im_noise))),inv_im(im_out)))
        
        # Ajouter une dimension pour les canaux pour chaque image
        im = np.expand_dims(im, axis=-1)

        # Sauvegarder les images avec haute qualité
        save_high_quality_image(im, title + str(i) + '.png')
        plt.close('all')




def print_loss(loss, label, i):
    plt.plot(loss)
    plt.savefig(label + str(i) + '.png')
    plt.close()





parser = argparse.ArgumentParser(description="DFBNet")
# NN information
parser.add_argument("--num_of_layers", type=int, default=15, help="Number of total layers")
parser.add_argument("--architecture", type=str, default='DFBNet', help="type of network to study.")
parser.add_argument("--name", type=str, required=True, help="where to save.")
# Training information
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=50, help="Training patch size")
parser.add_argument("--seed", type=int, default=1, help="Seed number")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--pretrained", type=int, default=0, help="Load pretrained model")
parser.add_argument("--outf", type=str, default="checkpoints", help='path to save trained NN')
parser.add_argument("--noise_lev", type=float, default=0.15, help='noise level')
parser.add_argument("--loss_name", type=str, default='l1', help='name training loss')
parser.add_argument("--variable_noise", type=int, default=1, help="variable level of noise or not")
parser.add_argument("--S", type=int, default=600, help='dimension of the output of L')
#m can be chosen in [[0,S]]
parser.add_argument("--m", type=int, default=300, help='form of the blocks of L')
parser.add_argument("--subsampling_ratio", type=float, default=0.5, help='subsampling ratio')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def train_net(loss_name='l1', pretrained=False, num_of_layers=20, batchSize=100, epochs=200):

    Test_mode = False # if True: no save to avoid overwrite
    print('... test mode: ', Test_mode)

    torch.manual_seed(opt.seed) # Reproducibility
    torch.backends.cudnn.benchmark = True 
    



    # * Paths 
    curpath = pathlib.Path().absolute()
    if 'ab3024' in str(curpath): # Arthur
        path_save = './Training/' + opt.name
    else:
        # Give your path for dataset and saving as above
        print('No path provided')

    create_dir(path_save)


    # * Define folders and files for saving
    # This folder will contain our training logs
    if opt.variable_noise == False:
        save_info_print = path_save+'/_' +opt.architecture+'_'+loss_name+'_patchsize='+str(opt.patchSize)+'_noise'+str(opt.noise_lev)+'_layers_'+str(num_of_layers)
    else:
        save_info_print = path_save+'/_' +opt.architecture+'_'+loss_name+'_patchsize='+str(opt.patchSize)+'_varnoise'+str(opt.noise_lev)+'_layers_'+str(num_of_layers)
    dirText  = save_info_print+'/_infos_training/'
    dir_checkpoint = save_info_print+'/_'+opt.outf+'/' 
    dir_checkpoint_pretrained = save_info_print+'/'+opt.outf+'/'  
    img_folder = save_info_print+'/_images/'
    
    create_dir(dirText)
    create_dir(dir_checkpoint)
    create_dir(img_folder)

    img_folder_test = img_folder + '/test/'
    img_folder_loss = img_folder + '/loss/'
    create_dir(img_folder_test)
    create_dir(img_folder_loss)

    if Test_mode is False:
        n_f = 'training_info.txt'
        textFile = open(dirText+'/'+n_f,"w+")
        textFile.write('Launching...\n')
        textFile.close()

        arthur_text = 'Arthur_info.txt'
        textFile = open(dirText+'/'+arthur_text,"w+")
        textFile.close()

    loader_train, loader_val, N = get_dataset(random_seed=opt.seed, bs=opt.batchSize)

    H, H_star, H_star_H, P_star, r, beta,iava = sparcity_op(N, opt.subsampling_ratio)
    #here H = H^* = H^* H


    model, net_name, clip_val, lr = get_model(opt.architecture, num_of_layers=num_of_layers)

    if pretrained == True:
        print('pretrained:', pretrained)
        try:
            pth_pret = os.path.join(dir_checkpoint_pretrained, net_name+'.pth')
            model = load_model(pth=pth_pret, net=model)
        except:
            print('WARNING: couldnt load pretrained')

    if loss_name == 'l1':
        criterion = nn.L1Loss(reduction='sum')
    elif loss_name == 'l2':
        criterion = nn.MSELoss(reduction='sum')

    if pretrained == True:
        weight_decay = 2e-5
    else:
        weight_decay = 2e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Original # 1e-3 for initial but after some comp 5e-4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    
    if Test_mode is False:
        textFile = open(dirText+'/'+n_f,"a")
        textFile.write('''                  
        Starting training {}:

        Pretrained: {}:
        Epochs: {}
        Batch size: {}
        Patch size: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}

        Algorithm info (Adam):
        loss: {}
        Learning rate: {}
        Weight decay : {}

        CUDA: {}
        '''.format(opt.architecture, pretrained, opt.epochs, opt.patchSize, opt.batchSize, len(loader_train),
        len(loader_val), opt.outf, loss_name, lr, weight_decay, cuda))
        textFile.close()

    list_loss = []

    for epoch in range(1, opt.epochs):

        torch.cuda.empty_cache()
        loss_tot, avg_snr_train, avg_snr_noise = 0, 0, 0
        model.train()

        #iterate on the batches :
        for i, data in enumerate(loader_train, 0):

            torch.cuda.empty_cache()

            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            data_true = data['image_true'].type(Tensor).to(torch.device('cuda')).detach() # Keep initial data in memory
            

            if opt.variable_noise:
                noise = torch.zeros((data_true.shape[0],r)).type(Tensor).to(torch.device('cuda'))
                stdN = np.random.uniform(0., 1., size=noise.shape[0])
                for _ in range(noise.shape[0]):
                    noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
                lambd = ((opt.noise_lev*torch.Tensor(stdN))**2).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)
            else:
                stdN = opt.noise_lev
                noise = opt.noise_lev*torch.randn_like(data_true.shape[0],r)
                lambd = (opt.noise_lev**2).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)

            data_noisy = H(data_true.view(data_true.shape[0], -1))+noise.to(torch.device('cuda'))
            

            bias = H_star(data_noisy).view(data_true.shape).to(torch.device('cuda'))
            #bias = H^* y
            beta = beta
            H_star_H = H_star_H

            x_in = bias.clone()
            x_out = model(x_in, beta, H_star_H, lambd).view(data_true.shape)
            #we initiate the algo with x_in = y the noisy image, and u_in = L(x_in).

            #data_TV = [approx_TV(data_noisy[0].cpu().numpy(), H_np,1000).reshape(28,28), approx_TV(data_noisy[1].cpu().numpy(), H_np,1000).reshape(28,28), approx_TV(data_noisy[2].cpu().numpy(), H_np,1000).reshape(28,28)]

            size_tot = data_true.shape[0]*data_true.shape[1]*data_true.shape[2]

            loss = criterion(data_true, x_out)/size_tot

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()

            loss_tot += loss.item()

            # with torch.no_grad():
            #     model.module.print_lip()

            with torch.no_grad():
                snr_in = snr_torch(noise, torch.zeros(noise.shape))
                snr_train = snr_torch(data_true, x_out)

                if np.isfinite(snr_train) and np.isfinite(snr_in):
                    avg_snr_train += snr_train
                    avg_snr_noise += snr_in


                if i%(max(len(loader_train)//4,1)) == 0: # Training info
                    infos_str = '[epoch {}][{}/{}] loss: {:.3e} SNRin: {:.2f} SNRout: {:.2f} \n'.format(epoch, i+1, len(loader_train), loss.item(), snr_in, snr_train)
                    print(infos_str)
            
            if Test_mode is True:
                if i==1:
                    break

        list_loss.append(loss_tot)
        writer.add_scalar("Loss/train", loss_tot, epoch)

        if Test_mode is False:
            textFile = open(dirText+'/'+n_f, "a")
            textFile.write(infos_str)
            textFile.close()
            # ----
            #if (epoch-1)%5 == 0:
            #    true, out, noisy = data_true[:3].clone().reshape(3,28,28).cpu().numpy(), x_out[:3].clone().detach().reshape(3,28,28).cpu().numpy(), data_noisy[:3].clone().reshape(3,28,28).cpu().numpy()
            #    imshowgray(true, noisy, out, data_TV, epoch, title=img_folder_test)

                
                
        if Test_mode is True:
            if epoch==1:
                break

        avg_snr_train /= (i+1)
        avg_snr_noise /= (i+1)

        if Test_mode is False:
            if epoch%50 == 0:
                torch.save(model.module.state_dict(), os.path.join(dir_checkpoint, net_name+'_epoch'+str(epoch)+'.pth'))

        # model.module.update_lip(data_noisy.shape)
        # model.module.print_lip()
        if Test_mode is False:
            torch.save(model.module.state_dict(), os.path.join(dir_checkpoint, net_name+'.pth'))
        scheduler.step()
    print_loss(list_loss, img_folder_loss, opt.epochs-1)
    
    loss_eval_net = 0
    for i, data in enumerate(loader_val,0):
        # Traitez les trois premiers éléments ici
        data_true = data['image_true'].type(Tensor).to(torch.device('cuda')).detach() # Keep initial data in memory

        if opt.variable_noise:
            noise = torch.zeros(data_true.shape[0],r).type(Tensor).to(torch.device('cuda'))
            stdN = np.random.uniform(0., 1., size=noise.shape[0])
            for _ in range(noise.shape[0]):
                noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
            lambd = ((opt.noise_lev*torch.Tensor(stdN))**2).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)
        else:
            stdN = opt.noise_lev
            noise = opt.noise_lev*torch.randn(data_true.shape[0],r)
            lambd = (opt.noise_lev**2).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)

        data_noisy = H(data_true.view(data_true.shape[0],-1))+noise.to(torch.device('cuda'))
        

        bias = H_star(data_noisy).view(data_true.shape).to(torch.device('cuda'))
        #bias = H^* y
        beta = beta
        H_star_H = H_star_H

        x_in = bias.clone()
        x_out = model(x_in, bias, beta, H_star_H, lambd).view(data_true.shape)
        
        size_tot = data_true.shape[0]*data_true.shape[1]*data_true.shape[2]
        
        loss_eval_net += float(criterion(data_true, x_out)/size_tot)
    
    loss_info = "Loss on the validation set of the convolution NN : "+ str(loss_eval_net/(i+1))
    textFile = open(dirText+'/'+arthur_text, "a")
    textFile.write(loss_info)
    textFile.close()  
    print(loss_info)

    W, sigma, tau = model.module.print_weights(beta)

    lips_classic = Lips_cte_Classic_conv(r, opt.num_of_layers, iava, W, tau, sigma)
    print(lips_classic)
    for j in range(min(3,data_true.shape[0])):
        true, out, noisy = data_true[j].clone().reshape(28,28).cpu().numpy(), x_out[j].clone().detach().reshape(28,28).cpu().numpy(), bias[j].clone().cpu().numpy()
        noisy = noisy.reshape(28,28)
        imshowgray(true, noisy, out, j, title=img_folder_test)



        
from PIL import Image
import scipy
import scipy.io

from numba import jit




if __name__ == "__main__":
    var_noise = bool(opt.variable_noise)
    train_net(loss_name=opt.loss_name, pretrained=opt.pretrained, num_of_layers=opt.num_of_layers, batchSize=opt.batchSize, epochs=opt.epochs)
    

writer.flush()
writer.close()