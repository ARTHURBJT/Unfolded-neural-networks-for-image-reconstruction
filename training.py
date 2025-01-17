# Training of an unfolded dual-FB network for Gaussian denoising

import argparse
import os
import pathlib
import random


import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



from im_class import get_dataset
from helpers import create_dir, save_several_images, snr_torch, get_image
from modelhelper import get_model, load_model
from sparcity_op import sparcity_op

from TV_training import approx_TV
from Lips_cte_Arthur import Lips_cte_Arthur, Lips_cte_Classic, Lips_cte_Classic_conv
from Jacob_lips import JacobianReg_l2
from math import sqrt

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


from tqdm import tqdm 
from operator_conv import get_operators


def inv_im(im):
    if len(im.shape) >1:
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                im[i,j] = 1-im[i,j]
        return im
    else:
        for j in range(im.shape[0]):
            im[j] = 1-im[j]
        return im


def save_high_quality_image(image, filename):
    plt.figure(figsize=(20, 20))  # Taille de la figure (20 x 20 pouces)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')  # Désactiver les axes
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)  # Sauvegarder avec haute qualité
    plt.close()



def print_list(list, label, name, lips = False):
    if lips:
        plt.figure()
        plt.plot(list[0],  label = 'lips Jacobian')
        plt.plot(list[1],  label = 'lips Classic')
        plt.plot(list[2],  label = 'lips Arthur')
        plt.xlabel('Epochs')  # Label for x-axis
        plt.ylabel(name)  # Label for y-axis
        plt.legend()
        plt.savefig(label + '.png')
        plt.close()
    else:
        plt.figure()
        n = len(opt.num_of_layers)
        for i,l in enumerate(list):
            plt.plot(l,  label = str(opt.num_of_layers[i%n])+'_layers')
        plt.xlabel('Epochs')  # Label for x-axis
        plt.ylabel(name)  # Label for y-axis
        plt.legend()
        plt.savefig(label + name+'.png')
        plt.close()



parser = argparse.ArgumentParser(description="DFBNet")
# NN information
parser.add_argument("--num_of_layers", type=list, default=[5,20,50], help="Number of total layers")
parser.add_argument("--architecture", type=str, default='DFBNet', help="type of network to study.")
parser.add_argument("--name", type=str, required=True, help="where to save.")
# Training information
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=50, help="Training patch size")
parser.add_argument("--seed", type=int, default=1, help="Seed number")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--pretrained", type=int, default=0, help="Load pretrained model")
parser.add_argument("--outf", type=str, default="checkpoints", help='path to save trained NN')
parser.add_argument("--noise_lev", type=float, default=0.15, help='noise level')
parser.add_argument("--loss_name", type=str, default='l1', help='name training loss')
parser.add_argument("--variable_noise", type=int, default=1, help="variable level of noise or not")
parser.add_argument("--S", type=list, default=[200], help='dimension of the output of L')
#m can be chosen in [[0,S]]
parser.add_argument("--block", type=int, default=0, help='should L be a block matrix?')
parser.add_argument("--m", type=int, default=100, help='form of the blocks of L')
parser.add_argument("--subsampling_ratio", type=float, default=0.5, help='subsampling ratio')
parser.add_argument("--conv", type=list, default=[0,2,1], help='should L be a conv layer?')
parser.add_argument("--lips", type=int, default=0, help='should we compute the lips?')
parser.add_argument("--autoname", type=int, default=1, help='')
parser.add_argument("--inpainting", type=int, default=1, help='')

opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



def train_net(loader_train, loader_val, N,conv,loss_name='l1', pretrained=False, num_of_layers=20, S = 600, batchSize=100, epochs=200):

    Test_mode = False # if True: no save to avoid overwrite
    print('... test mode: ', Test_mode)

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # * Paths 
    curpath = pathlib.Path().absolute()
    if 'ab3024' in str(curpath): # Arthur
        if opt.autoname:
            if opt.subsampling_ratio == 1.:
                if conv == 2:
                    path_save = './Training_31_08/denoiser/conv'
                elif conv == 1:
                    path_save = './Training_31_08/denoiser/conv_multi_layers'
                else:
                    path_save = './Training_31_08/denoiser/lin'
            else:
                if conv == 2:
                    path_save = './Training_31_08/inpainting/conv'
                elif conv == 1:
                    path_save = './Training_31_08/inpainting/conv_multi_layers'
                else:
                    path_save = './Training_31_08/inpainting/lin'
        else:
            path_save = './Training_31_08/' + opt.name
    else:
        # Give your path for dataset and saving as above
        print('No path provided')

    create_dir(path_save)


    # * Define folders and files for saving
    # This folder will contain our training logs
    if opt.variable_noise == False:
        save_info_print = path_save+'/_' +opt.architecture+'_'+loss_name+'_patchsize='+str(opt.patchSize)+'_noise'+str(opt.noise_lev)+'_layers_'+str(num_of_layers)+'_S_'+str(S)
    else:
        save_info_print = path_save+'/_' +opt.architecture+'_'+loss_name+'_patchsize='+str(opt.patchSize)+'_varnoise'+str(opt.noise_lev)+'_layers_'+str(num_of_layers)+'_S_'+str(S)
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

    if opt.inpainting:
        H, H_star, H_star_H, P_star, r, beta, iava = sparcity_op(opt.seed, N, opt.subsampling_ratio)
        #here H = H^* = H^* H
    else:
        type_op = 'torch_circular_conv'
        blur_choice = 'blur_1'
        kernel_id = blur_choice +'.mat'

        H,H_star, H_star_H,beta = get_operators(type_op = type_op, kernel_id = kernel_id, cross_cor = True)
        
        r = 28*28
        #vp...
        #beta = torch.max(vp)

    model, net_name, clip_val, lr = get_model(opt.seed, opt.architecture, N, S, conv, opt.m, r, P_star, opt.block, num_of_layers=num_of_layers)


    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # # Print the number of parameters
    # print(f'The model has {count_parameters(model)} trainable parameters')


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
        weight_decay = 2e-5*0
    else:
        weight_decay = 2e-5*0

    
    if conv == 0:
        params_with_different_lrs = [
            {'params': model.module.L.parameters(), 'lr': 0.01},
            {'params': model.module.linlist.parameters(), 'lr': 0.001}
        ]
    elif conv == 1:
        params_with_different_lrs = [{'params': model.module.linlist[_].conv.parameters(), 'lr': 0.001} for _ in range(1,num_of_layers+1)]+[{'params': model.module.linlist[_].log_sigma, 'lr': 0.01} for _ in range(1,num_of_layers+1)]
        weight_decay = 2e-4
    else:
        params_with_different_lrs = [
            {'params': model.module.L.parameters(), 'lr': 0.001},
            {'params': model.module.linlist.parameters(), 'lr': 0.01}
        ]
    #     

    optimizer = optim.Adam(params_with_different_lrs, lr=0.001, weight_decay=weight_decay)  # Original # 1e-3 for initial but after some comp 5e-4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    
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

    list_loss_train = []
    list_loss_val = []

    lips_L_list = []
    sigma_list = [[] for _ in range(num_of_layers)]
    # lips_arthur_list = []
    # lips_classic_list = []
    # lips_jac_list = []

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    for epoch in range(1, opt.epochs+1):

        loss_tot = 0
        #iterate on the batches :
        for i, data in tqdm(enumerate(loader_train, 0), desc=f"Epoch:{epoch}, training",ncols = 100):
            
            if conv!=0:
                model.module.update_lip((1,28,28))
            else:
                model.module.update_lip(N)
            
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            model.zero_grad()
            model.train()


            #model.module.L.weight.data = torch.clamp(model.module.L.weight.data, min = -1/(2*sqrt(N*S)), max = 1/(2*sqrt(N*S)))


            data_true = data['image_true'].type(Tensor).to(torch.device('cuda')).detach() # Keep initial data in memory

            if opt.variable_noise:
                noise = torch.zeros(data_true.shape[0],r).type(Tensor).to(torch.device('cuda'))
                stdN = np.random.uniform(0., 1., size=noise.shape[0])
                for _ in range(noise.shape[0]):
                    noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
                lambd = ((opt.noise_lev*torch.Tensor(stdN))**2).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
            else:
                noise = opt.noise_lev*torch.randn_like(data_true).to(torch.device('cuda'))
                lambd = (opt.noise_lev**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))

            data_noisy = (H(data_true)+noise).to(torch.device('cuda'))
            
            y = H_star(data_noisy)

            if conv != 0:
                lambd.unsqueeze_(1).unsqueeze_(1).type(Tensor)
            x_out = model(y, beta, H_star_H, lambd)
            #we initiate the algo with x_in = y the noisy image, and u_in = L(x_in).

            size_tot = data_true.size()[0]*data_true.size()[1]

            loss = criterion(data_true, x_out)/size_tot 
            #loss = criterion(data_true, x_out)/size_tot + nn.ReLU()(torch.norm(model.module.L.weight) -1)

            loss.backward()
            #model.module.L.weight.data = torch.clamp(model.module.L.weight.data, min = -1/(2*sqrt(N*S)), max = 1/(2*sqrt(N*S)))


            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()

            loss_tot += loss.item()

            # with torch.no_grad():
            #     model.module.print_lip()

            with torch.no_grad():
                snr_noise = snr_torch(data_true, y)
                snr_train = snr_torch(data_true, x_out)

                if (i+1)%(max(len(loader_train)//4,1)) == 0: # Training info
                    infos_str = '[epoch {}][{}/{}] loss: {:.3e} SNRin: {:.2f} SNRout: {:.2f} \n'.format(epoch, i+1, len(loader_train), loss.item(), snr_noise, snr_train)
                    print(infos_str)

            if Test_mode is True:
                if i==1:
                    break
        
        with torch.no_grad():
            if conv!=0:
                model.module.update_lip((1,28,28), eval = True)
            else:
                model.module.update_lip(N, eval = True)
            loss_tot = loss_tot/(i+1)
            loss_val_net = 0
            for i, data in tqdm(enumerate(loader_val, 0), desc=f"Epoch:{epoch}, validation",ncols = 100):
                data_true = data['image_true'].type(Tensor).to(torch.device('cuda')).detach() # Keep initial data in memory
                
                if opt.variable_noise:
                    noise = torch.zeros(data_true.shape[0],r).type(Tensor).to(torch.device('cuda'))
                    stdN = np.random.uniform(0., 1., size=noise.shape[0])
                    for _ in range(noise.shape[0]):
                        noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
                    lambd = ((opt.noise_lev*torch.Tensor(stdN))**2).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
                else:
                    noise = opt.noise_lev*torch.randn_like(data_true).to(torch.device('cuda'))
                    lambd = (opt.noise_lev**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))

                data_noisy = (H(data_true)+noise).to(torch.device('cuda'))
                #H^*H = H^2 = H
                y = H_star(data_noisy)

                if conv!=0:
                    lambd.unsqueeze_(1).unsqueeze_(1).type(Tensor)
                x_out = model(y, beta, H_star_H, lambd, eval = True)
                    
                size_tot = data_true.size()[0]*data_true.size()[1]
                loss_val_net += float(criterion(data_true, x_out)/size_tot)
            loss_val_net = loss_val_net/(i+1)

            list_loss_train.append(loss_tot)
            list_loss_val.append(loss_val_net)

            if conv != 2:
                w, sigma_, tau_ = model.module.print_weights(beta)
            else:
                L,L_t, sigma_, tau_ = model.module.print_weights(beta)
            lips_L = sqrt(model.module.lip2)
        #     #lips_L_info = "Spectral norm of the matrix L : "+str(lips_L)
        #     #print(lips_L_info)
            lips_L_list.append(lips_L)
            for _ in range(num_of_layers):
                sigma_list[_].append(sigma_[_].item())

        #     if opt.block:
        #         #Lipschitz constants : 
        #         B_1 = w[:opt.m,:r]
        #         B_2 = w[opt.m:,r:]
        #         lips_arthur = Lips_cte_Arthur(num_of_layers,N,r,tau_,sigma_,B_1,B_2).item()
        #         #lips_arthur_info = "Lipschitz constant arthur of the NN : "+str(lips_arthur)
        #         #print(lips_arthur_info)
        #         lips_arthur_list.append(lips_arthur)

        #     lips_cte_classic = Lips_cte_Classic(N, S, num_of_layers, r, iava, w, tau_, sigma_).item()
        #     #lips_classic_info = "Lipschitz constant classic of the NN : "+str(lips_cte_classic)
        #     #print(lips_classic_info)
        #     lips_classic_list.append(lips_cte_classic)

        # lips_cte_jacob = 0
        # for i, data in enumerate(loader_val, 0):
        #     torch.cuda.empty_cache()

        #     model.zero_grad()
            
        #     data_true = data['image_true'].type(Tensor).to(torch.device('cuda')) # Keep initial data in memory
            
        #     if opt.variable_noise:
        #         noise = torch.zeros(data_true.shape[0],r).type(Tensor).to(torch.device('cuda'))
        #         stdN = np.random.uniform(0., 1., size=noise.shape[0])
        #         for _ in range(noise.shape[0]):
        #             noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
        #         lambd = ((opt.noise_lev/2)**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
        #     else:
        #         noise = opt.noise_lev*torch.randn_like(data_true).to(torch.device('cuda'))
        #         lambd = ((opt.noise_lev/2)**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
            
        #     data_noisy = (H(data_true)+noise).to(torch.device('cuda'))

            
        #     y = H_star(data_noisy)
        #     y = y.to(torch.device('cuda')).requires_grad_()
        #     x_out = model(y, beta, H_star_H, lambd)

            
        #     lips_cte_jacob = max(JacobianReg_l2().Lips(y,x_out),lips_cte_jacob)
        #     y.detach_()
        #     x_out.detach_()
        #     model.zero_grad()

        #     if i>2:
        #         break

        # #lips_jacob_info = "Lipschitz constant with the Jacobian method of the NN : "+str(lips_cte_jacob)
        # #print(lips_jacob_info)
        # lips_jac_list.append(lips_cte_jacob)

        # torch.cuda.empty_cache()
        # model.zero_grad()

            
        writer.add_scalar("Loss/test", loss_tot, epoch)
        writer.add_scalar("Loss/val", loss_val_net, epoch)
        writer.add_scalar("Lips_L", lips_L, epoch)
        # #writer.add_scalar("Lips_net", lips_arthur, epoch)
        writer.add_scalar("Sigma_0", sigma_[0].item(), epoch)
        writer.add_scalar("Sigma_1", sigma_[num_of_layers-1].item(), epoch)

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

        if Test_mode is False:
            if epoch%50 == 0:
                torch.save(model.module.state_dict(), os.path.join(dir_checkpoint, net_name+'_epoch'+str(epoch)+'.pth'))

        # model.module.update_lip(data_noisy.shape)
        # model.module.print_lip()
        if Test_mode is False:
            torch.save(model.module.state_dict(), os.path.join(dir_checkpoint, net_name+'.pth'))
        scheduler.step()



    with torch.no_grad():
        loss_val_net = 0
        loss_val_TV_max = 0
        loss_val_TV = 0
        avg_snr_val, avg_snr_noise = 0, 0
        avg_snr_TV, avg_snr_TV_max = 0,0
        avg_it = 0
        for i, data in enumerate(loader_val, 0):
            data_true = data['image_true'].type(Tensor).to(torch.device('cuda')).detach() # Keep initial data in memory
            
            if opt.variable_noise:
                noise = torch.zeros(data_true.shape[0],r).type(Tensor).to(torch.device('cuda'))
                stdN = np.random.uniform(0., 1., size=noise.shape[0])
                for _ in range(noise.shape[0]):
                    noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
                lambd = ((opt.noise_lev*torch.Tensor(stdN))**2).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
            else:
                noise = opt.noise_lev*torch.randn_like(data_true).to(torch.device('cuda'))
                lambd = (opt.noise_lev**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))

            data_noisy = (H(data_true)+noise).to(torch.device('cuda'))
            #H^*H = H^2 = H
            y = H_star(data_noisy)
            if conv!=0:
                lambd.unsqueeze_(1).unsqueeze_(1).type(Tensor)
            x_out = model(y, beta, H_star_H, lambd)

            # if S == opt.S[0] and conv == 0:
            #     #reshap = data_true.reshape(data_true.shape[0],28,28)[:,1:27,1:27].reshape(data_true.shape[0],26,26)
            #     reshap = data_true
            #     x_TV,it = approx_TV(data_noisy, lambd, H, H_star,num_of_layers)
            #     #x_TV = x_TV.reshape(data_true.shape[0],28,28)[:,1:27,1:27].reshape(data_true.shape[0],26,26)
            #     x_TV = x_TV
            #     x_TV_max,it = approx_TV(data_noisy, lambd, H, H_star,1000)
            #     #x_TV_max = x_TV_max.reshape(data_true.shape[0],28,28)[:,1:27,1:27].reshape(data_true.shape[0],26,26)
            #     x_TV_max = x_TV_max
            #     loss_val_TV_max += float(criterion(reshap, x_TV_max)/size_tot)
            #     loss_val_TV += float(criterion(reshap, x_TV)/size_tot)
            #     snr_TV = snr_torch(reshap, x_TV)
            #     snr_TV_max = snr_torch(reshap, x_TV_max)
            #     avg_snr_TV += snr_TV
            #     avg_snr_TV_max += snr_TV_max
            #     avg_it += it
                
            size_tot = data_true.size()[0]*data_true.size()[1]
            loss_val_net += float(criterion(data_true, x_out)/size_tot)
            
            snr_noise = snr_torch(data_true, y)
            snr_val = snr_torch(data_true, x_out)

            if np.isfinite(snr_val) and np.isfinite(snr_noise):
                avg_snr_val += snr_val
                avg_snr_noise += snr_noise

        loss_val_net = loss_val_net/(i+1)
        loss_val_TV_max = loss_val_TV_max/(i+1)
        loss_val_TV = loss_val_TV/(i+1)
        avg_snr_val /= (i+1)
        avg_snr_noise /= (i+1)
        avg_snr_TV /= (i+1)
        avg_snr_TV_max /= (i+1)
        avg_it /= (i+1)

    im = []
    for j in range(min(3,data_true.shape[0])):
        true, out, noisy = inv_im(data_true[j].clone().reshape(28,28).cpu().numpy()), inv_im(x_out[j].clone().detach().reshape(28,28).cpu().numpy()), inv_im(y[j].clone().reshape(28,28).cpu().numpy())
        if conv == 0:
            im.append(np.concatenate((np.clip(true,0,1),np.clip(noisy,0,1),np.clip(out,0,1)), axis=1))
        else:
            im.append(np.clip(out,0,1))
        
        #imshowgray(true, noisy, out, j, title=img_folder_test + '_NN')

        # if S == opt.S[0] and conv == 0:
        #     TV, TV_max = x_TV[j].clone().reshape(28,28).cpu().numpy(), x_TV_max[j].clone().reshape(28,28).cpu().numpy()
        #     TV = inv_im(TV)
        #     TV_max = inv_im(TV_max)
        #     imshowgray(true, noisy, TV, j, title=img_folder_test + '_TV')
        #     imshowgray(true, noisy, TV_max, j, title=img_folder_test + '_TV_max')

    loss_NN_info = "Loss on the validation set of the NN after training : "+str(loss_val_net)
    #print(loss_NN_info)
    loss_TV_info = "Loss on the validation set of the TV algorithm with the same number of iteration than the NN : "+str(loss_val_TV)
    #print(loss_TV_info)
    loss_TV_info_max = "Loss on the validation set of the TV algorithm when it converges : "+str(loss_val_TV_max)
    #print(loss_TV_info_max)
    
    textFile = open(dirText+'/'+arthur_text, "a")
    textFile.write(loss_NN_info+'\n')
    textFile.write(loss_TV_info+'\n')
    textFile.write(loss_TV_info_max+'\n')
    textFile.write('snr of the noise :' + str(avg_snr_noise) + '\n')
    textFile.write('snr of the reconstruct image, NN :' + str(avg_snr_val)+'\n')
    textFile.write('snr of the reconstruct image, TV :' + str(avg_snr_TV)+'\n')
    textFile.write('snr of the reconstruct image, TV CV :' + str(avg_snr_TV_max)+'\n')
    textFile.write('average iteration of TV to CV :' + str(avg_it)+'\n')
    textFile.close()

    if opt.lips:
        #Lipschitz constants : 
        if conv == 0:
            w, sigma_, tau_ = model.module.print_weights(beta)
        elif conv == 2:
            L, L_t, sigma_, tau_ = model.module.print_weights(beta)
        if conv!=0:
            model.module.update_lip((1,28,28))
        else:
            model.module.update_lip(N)
        lips_L = sqrt(model.module.lip2)
        lips_L_info = "Spectral norm of the matrix L : "+str(lips_L)
        print(lips_L_info)

        textFile = open(dirText+'/'+arthur_text, "a")
        textFile.write(lips_L_info+'\n')
        textFile.close()

        lips_arthur = 0
        # if opt.block:
        #     B_1 = w[:opt.m,:r]
        #     B_2 = w[opt.m:,r:]
        if conv == 0:
            lips_arthur = Lips_cte_Arthur(num_of_layers,N,r,tau_,sigma_,w).item()
            lips_arthur_info = "Lipschitz constant arthur of the NN : "+str(lips_arthur)

            print(lips_arthur_info)
            textFile = open(dirText+'/'+arthur_text, "a")
            textFile.write(lips_arthur_info+'\n')
            textFile.close()

        if conv == 0:
            lips_cte_classic = Lips_cte_Classic(N, S, num_of_layers, r, iava, w, tau_, sigma_).item()
        if conv == 2:
            lips_cte_classic = Lips_cte_Classic_conv(N, (1,64,28,28), num_of_layers, iava, L, L_t, tau_, sigma_).item()

        lips_classic_info = "Lipschitz constant classic of the NN : "+str(lips_cte_classic)

        print(lips_classic_info)
        textFile = open(dirText+'/'+arthur_text, "a")
        textFile.write(lips_classic_info+'\n')
        textFile.close()


        lips_cte_jacob = 0
        for i, data in enumerate(loader_val, 0):
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            model.zero_grad()
            model.train()
            
            data_true = data['image_true'].type(Tensor).to(torch.device('cuda')).requires_grad_() # Keep initial data in memory
            
            if opt.variable_noise:
                noise = torch.zeros(data_true.shape[0],r).type(Tensor).to(torch.device('cuda'))
                stdN = np.random.uniform(0., 1., size=noise.shape[0])
                for _ in range(noise.shape[0]):
                    noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor).to(torch.device('cuda'))
                lambd = ((opt.noise_lev/2)**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
            else:
                noise = opt.noise_lev*torch.randn_like(data_true).to(torch.device('cuda'))
                lambd = ((opt.noise_lev/2)**2 * torch.ones(noise.shape[0])).unsqueeze_(1).type(Tensor).to(torch.device('cuda'))
            
            data_noisy = (H(data_true)+noise).to(torch.device('cuda'))

            y = H_star(data_noisy)
            if conv!=0:
                lambd.unsqueeze_(1).unsqueeze_(1).type(Tensor)
            x_out = model(y, beta, H_star_H, lambd)
            
            lips_cte_jacob = max(JacobianReg_l2().Lips(y,x_out),lips_cte_jacob)
            
            # data_TV_max = [approx_TV(data_noisy[0].cpu().numpy(), H, H_star,1000).reshape(28,28), approx_TV(data_noisy[1].cpu().numpy(), H, H_star,1000).reshape(28,28), approx_TV(data_noisy[2].cpu().numpy(), H, H_star,1000).reshape(28,28)]
            # data_TV = [approx_TV(data_noisy[0].cpu().numpy(), H, H_star,opt.num_of_layers).reshape(28,28), approx_TV(data_noisy[1].cpu().numpy(), H, H_star,opt.num_of_layers).reshape(28,28), approx_TV(data_noisy[2].cpu().numpy(), H, H_star,opt.num_of_layers).reshape(28,28)]

            if i>-1:
                break
        lips_jacob_info = "Lipschitz constant with the Jacobian method of the NN : "+str(lips_cte_jacob)

        print(lips_jacob_info)
        textFile = open(dirText+'/'+arthur_text, "a")
        textFile.write(lips_jacob_info+'\n')
        textFile.close()



        # l = [Lips_cte_Classic(N, S, num_of_layers, r, iava, 1*k*w/10, tau_, sigma_).item() for k in range(1,11)]
        # l2 = [lips_L * 1*k/10  for k in range(1,11)]
        # plt.figure()
        # plt.plot(l2,l)
        # plt.xlabel('Operator norm of L')  # Label for x-axis
        # plt.ylabel('alpha-layers')  # Label for y-axis
        # plt.title('Layer decomposition Lipschitz constant depending on the norm of L')  # Title for the plot
        # plt.legend()
        # plt.savefig(img_folder_loss+'lips_norm_L' + '.png')
        # plt.close()


        return im,list_loss_train, list_loss_val, img_folder_loss, lips_cte_classic, lips_cte_jacob, lips_arthur, lips_L_list, sigma_list
    
    else:
        return im, list_loss_train, list_loss_val, img_folder_loss, lips_L_list, sigma_list


        
from PIL import Image
import scipy
import scipy.io

from numba import jit




def print_loss(loss_train, loss_val, label_, names, epoch):
    plt.figure()
    for i,l in enumerate(loss_train):
        plt.plot(l, label = 'Training loss'+'_'+str(names[i])+'_layers')
    #for i,l in enumerate(loss_val):
    #    plt.plot(l, label = 'Validation loss'+'_'+str(names[i])+'_layers')
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('Loss')  # Label for y-axis
    plt.title('Training and Validation Loss')  # Title for the plot
    plt.legend()
    plt.savefig(label_+"_all_training_loss" + '.png')
    plt.close()



if __name__ == "__main__":
    var_noise = bool(opt.variable_noise)
    list_loss_train, list_loss_val, lips_arthur_list, lips_jac_list, lips_classic_list = [],[],[],[],[]
    names = []
    lips_layers_L_list = []

    loader_train, loader_val, N = get_dataset(random_seed=opt.seed, bs=opt.batchSize)
    
    for num in opt.num_of_layers:
        for conv in opt.conv:
            for S in opt.S:
                np.random.seed(opt.seed) # Reproducibility
                torch.manual_seed(opt.seed) # Reproducibility
                if conv == 0 and not opt.lips:
                    im, train, val, img_folder_loss, lips_L_list, sigma_list = train_net(loader_train, loader_val, N, conv,loss_name=opt.loss_name, pretrained=opt.pretrained, num_of_layers=num, S = S, batchSize=opt.batchSize, epochs=opt.epochs)
                    if S == opt.S[-1]:
                        list_loss_train.append(train)
                        list_loss_val.append(val)
                        names.append(num)
                    plt.figure()
                    plt.plot(train, label = 'Training loss'+'_'+str(num)+'_layers')
                    plt.plot(val, label = 'Validation loss'+'_'+str(num)+'_layers')
                    plt.xlabel('Epochs')  # Label for x-axis
                    plt.ylabel('Loss')  # Label for y-axis
                    plt.title('Training and Validation Loss')  # Title for the plot
                    plt.legend()
                    plt.savefig(img_folder_loss+"loss_"+str(num) + '_layers_S=' + str(S)+ '.png')
                    plt.close()

                    for _ in [0,1,num-1]:
                        print_list([sigma_list[_]], img_folder_loss, "sigma"+str(_))
                    print_list([lips_L_list], img_folder_loss, "lips_L")

                    im1 = im[0]
                    im2 = im[1]
                    im3 = im[2]

                elif not opt.lips:
                    #we want to try different S only for the linear NN
                    if S == opt.S[-1]:
                        im, train, val, img_folder_loss, lips_L_list, sigma_list = train_net(loader_train, loader_val, N, conv,loss_name=opt.loss_name, pretrained=opt.pretrained, num_of_layers=num, S = S, batchSize=opt.batchSize, epochs=opt.epochs)
                        list_loss_train.append(train)
                        list_loss_val.append(val)
                        names.append(num)

                        plt.figure()
                        plt.plot(train, label = 'Training loss'+'_'+str(num)+'_layers')
                        plt.plot(val, label = 'Validation loss'+'_'+str(num)+'_layers')
                        plt.xlabel('Epochs')  # Label for x-axis
                        plt.ylabel('Loss')  # Label for y-axis
                        plt.title('Training and Validation Loss')  # Title for the plot
                        plt.legend()
                        plt.savefig(img_folder_loss+"loss_"+str(num) + '_layers' + '.png')
                        plt.close()

                        for _ in [0,1,num-1]:
                            print_list([sigma_list[_]], img_folder_loss, "sigma"+str(_))
                        print_list([lips_L_list], img_folder_loss, "lips_L")

                        im1 = np.concatenate((im1,im[0]),axis = 1)
                        im2 = np.concatenate((im2,im[1]),axis = 1)
                        im3 = np.concatenate((im3,im[2]),axis = 1)


                else:#opt.lips == True
                    im, train, val, img_folder_loss, cla, jac, art, lips_L_list, sigma_list = train_net(loader_train, loader_val, N, conv,loss_name=opt.loss_name, pretrained=opt.pretrained, num_of_layers=num, S=S, batchSize=opt.batchSize, epochs=opt.epochs)
                    if S == opt.S[-1]:
                        list_loss_train.append(train)
                        list_loss_val.append(val)
                        lips_arthur_list.append(art)
                        lips_jac_list.append(jac)
                        lips_classic_list.append(cla)
                        lips_layers_L_list.append(lips_L_list[-1])
                        names.append(num)
                
                    plt.figure()
                    plt.plot(train, label = 'Training loss'+'_'+str(num)+'_layers')
                    plt.plot(val, label = 'Validation loss'+'_'+str(num)+'_layers')
                    plt.xlabel('Epochs')  # Label for x-axis
                    plt.ylabel('Loss')  # Label for y-axis
                    plt.title('Training and Validation Loss')  # Title for the plot
                    plt.legend()
                    plt.savefig(img_folder_loss+"loss_"+str(num) + '_layers' + '.png')
                    plt.close()

                    for _ in [0,num-1]:
                        print_list([sigma_list[_]], img_folder_loss, "sigma"+str(_))
                    print_list([lips_L_list], img_folder_loss, "lips_L")

        im1 = np.expand_dims(im1, axis=-1)
        im2 = np.expand_dims(im2, axis=-1)
        im3 = np.expand_dims(im3, axis=-1)

        save_high_quality_image(im1, img_folder_loss+'example_im_1.png')
        save_high_quality_image(im2, img_folder_loss+'example_im_2.png')
        save_high_quality_image(im3, img_folder_loss+'example_im_3.png')


    # print_loss(list_loss_train, list_loss_val, img_folder_loss, names, opt.epochs-1)

    # if opt.block:
    #     plt.figure()
    #     plt.plot(opt.num_of_layers, lips_classic_list, label = 'Classical')
    #     plt.plot(opt.num_of_layers, lips_jac_list, label = 'Jacobian')
    #     plt.plot(opt.num_of_layers, lips_arthur_list, label = 'Arthur')
    #     plt.xlabel('Number of layers')  # Label for x-axis
    #     plt.ylabel('Lipschitz constant')  # Label for y-axis
    #     plt.title('The different Lipschitz constants')  # Title for the plot
    #     plt.legend()
    #     plt.savefig(img_folder_loss + "lips_cla_jac_art" + '_'+ '.png')
    #     plt.close()
    if opt.lips:
        if conv == 0:
            plt.figure()
            plt.yscale('log')
            plt.plot(opt.num_of_layers, lips_classic_list, label = 'Layer decomposition')
            plt.plot(opt.num_of_layers, lips_arthur_list, label = 'Approximation')
            plt.plot(opt.num_of_layers, lips_jac_list, label = 'Jacobian')
            plt.xlabel('Number of layers')  # Label for x-axis
            plt.ylabel('Lipschitz constant')  # Label for y-axis
            plt.title('The different Lipschitz constants')  # Title for the plot
            plt.legend()
            plt.savefig(img_folder_loss + "lips_cla_jac" + '_'+ '.png')
            plt.close()

            plt.figure()
            plt.yscale('log')
            plt.plot(opt.num_of_layers, lips_arthur_list, label = 'Approximation')
            plt.xlabel('Number of layers')  # Label for x-axis
            plt.ylabel('Lipschitz constant')  # Label for y-axis
            plt.title('Orthodiagonalisation Lipschitz constants')  # Title for the plot
            plt.legend()
            plt.savefig(img_folder_loss + "approx" + '_'+ '.png')
            plt.close()
        plt.figure()
        plt.plot(opt.num_of_layers, lips_layers_L_list, label = 'L')
        plt.xlabel('Number of layers')  # Label for x-axis
        plt.ylabel('Lipschitz constant')  # Label for y-axis
        plt.title('L Lipschitz constants')  # Title for the plot
        plt.legend()
        plt.savefig(img_folder_loss + "lips_L_layers" + '_'+ '.png')
        plt.close()

        plt.figure()
        plt.yscale('log')
        plt.plot(opt.num_of_layers, lips_classic_list, label = 'Layer decomposition')
        plt.xlabel('Number of layers')  # Label for x-axis
        plt.ylabel('Lipschitz constant')  # Label for y-axis
        plt.title('Layers decomposition Lipschitz constants')  # Title for the plot
        plt.legend()
        plt.savefig(img_folder_loss + "lay" + '_'+ '.png')
        plt.close()

        plt.figure()
        plt.yscale('log')
        plt.plot(opt.num_of_layers, lips_jac_list, label = 'Jacobian')
        plt.xlabel('Number of layers')  # Label for x-axis
        plt.ylabel('Lipschitz constant')  # Label for y-axis
        plt.title('Jacobian Lipschitz constants')  # Title for the plot
        plt.legend()
        plt.savefig(img_folder_loss + "jac" + '_'+ '.png')
        plt.close()

writer.flush()
writer.close()




        