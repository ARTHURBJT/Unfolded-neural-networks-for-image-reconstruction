# Training of an unfolded dual-FB network for Gaussian denoising

import argparse
import os
import pathlib

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from im_class_1 import get_dataset
from helpers import create_dir, save_several_images, snr_torch, get_image
from modelhelper_1 import get_model, load_model

parser = argparse.ArgumentParser(description="DFBNet")
# NN information
parser.add_argument("--n_ch", type=int, default=3, help="number of channels")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--num_of_features", type=int, default=64, help="Convolution dimension")
parser.add_argument("--architecture", type=str, default='DFBNet', help="type of network to study.")
# Training information
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=50, help="Training patch size")
parser.add_argument("--seed", type=int, default=1, help="Seed number")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--pretrained", type=int, default=0, help="Load pretrained model")
parser.add_argument("--outf", type=str, default="checkpoints", help='path to save trained NN')
parser.add_argument("--noise_lev", type=float, default=0.05, help='noise level')
parser.add_argument("--loss_name", type=str, default='l1', help='name training loss')
parser.add_argument("--variable_noise", type=int, default=1, help="variable level of noise or not")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def train_net(loss_name='l1', pretrained=False, num_of_features=64, num_of_layers=20, batchSize=100, epochs=200):

    Test_mode = False # if True: no save to avoid overwrite
    print('... test mode: ', Test_mode)

    torch.manual_seed(0) # Reproducibility
    torch.backends.cudnn.benchmark = True 
    



    # * Paths 
    curpath = pathlib.Path().absolute()
    if 'ar50' in str(curpath): # Audrey 
        path_dataset = '../../../../scratch/ar50/dataset/imagenet_test/' 
        print(path_dataset)
        path_BSDS = None
        pattern_red = '*.JPEG'
        # path_save = './'
        path_save = '../../OneDrive/CODES/DFBNet-PnP-results/Training_'+opt.architecture+'/'
    else:
        # Give your path for dataset and saving as above
        print('No path provided')

    create_dir(path_save)


    # * Define folders and files for saving
    # This folder will contain our training logs
    if opt.variable_noise == False:
        save_info_print = opt.architecture+'_'+loss_name+'_patchsize='+str(opt.patchSize)+'_noise'+str(opt.noise_lev)+'_feat_'+str(num_of_features)+'_layers_'+str(num_of_layers)
    else:
        save_info_print = opt.architecture+'_'+loss_name+'_patchsize='+str(opt.patchSize)+'_varnoise'+str(opt.noise_lev)+'_feat_'+str(num_of_features)+'_layers_'+str(num_of_layers)
    dirText  = path_save+'infos_training/' +save_info_print+'/'
    dir_checkpoint = path_save+opt.outf+'/'+save_info_print+'/' 
    dir_checkpoint_pretrained = path_save+opt.outf+'/'+save_info_print+ '/'  
    img_folder = path_save+'images/'+save_info_print+'/'
    
    create_dir(dirText)
    create_dir(dir_checkpoint)
    create_dir(img_folder)

    if Test_mode is False:
        n_f = 'training_info.txt'
        textFile = open(dirText+'/'+n_f,"w+")
        textFile.write('Launching...\n')
        textFile.close()


    loader_train, loader_val = get_dataset(path_dataset=path_dataset, path_red_dataset=path_BSDS, patchSize=opt.patchSize, red=True, red_size=10, channels=opt.n_ch, bs=opt.batchSize, pattern_red=pattern_red, num_workers=6)

    model, net_name, clip_val, lr = get_model(opt.architecture, n_ch=opt.n_ch, features=num_of_features, num_of_layers=num_of_layers)

    if pretrained == True:
        print('pretrained:', pretrained)
        try:
            pth_pret = os.path.join(dir_checkpoint_pretrained, net_name+'.pth')
            model = load_model(pth=pth_pret, net=model, n_ch=opt.n_ch, features=num_of_features, num_of_layers=num_of_layers)
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

    for epoch in range(1, opt.epochs):

        torch.cuda.empty_cache()
        loss_tot, avg_snr_train, avg_snr_noise = 0, 0, 0
        model.train()

        for i, data in enumerate(loader_train, 0):

            torch.cuda.empty_cache()

            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            
            data_true = Variable(data['image_true'].type(Tensor), requires_grad=False)  # Keep initial data in memory
            

            if opt.variable_noise:
                noise = torch.zeros(data_true.shape).type(Tensor)
                stdN = np.random.uniform(0., 1., size=noise.shape[0])
                for _ in range(noise.shape[0]):
                    noise[_, ...] = opt.noise_lev*stdN[_]*torch.randn(noise[_,...].shape).type(Tensor)
                ths_map = ((opt.noise_lev*torch.Tensor(stdN))**2).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)
            else:
                stdN = opt.noise_lev
                noise = opt.noise_lev*torch.randn_like(data_true)
                ths_map = (opt.noise_lev**2).unsqueeze_(1).unsqueeze_(1).unsqueeze_(1).type(Tensor)

            data_noisy = data_true+noise

            out = model(data_noisy, data_noisy, ths_map)
            est_noise = data_noisy-out

            size_tot = data_true.size()[0]*2*data_true.size()[-1]**2

            loss = criterion(est_noise, noise)/size_tot

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()

            loss_tot += loss.item()

            # with torch.no_grad():
            #     model.module.print_lip()

            with torch.no_grad():
                snr_in = snr_torch(data_true, data_noisy)
                snr_train = snr_torch(data_true, out)

                if np.isfinite(snr_train) and np.isfinite(snr_in):
                    avg_snr_train += snr_train
                    avg_snr_noise += snr_in


                if i%50 == 0: # Training info
                    infos_str = '[epoch {}][{}/{}] loss: {:.3e} SNRin: {:.2f} SNRout: {:.2f} \n'.format(epoch, i+1, len(loader_train), loss.item(), snr_in, snr_train)
                    print(infos_str)
                    if Test_mode is False:
                        textFile = open(dirText+'/'+n_f, "a")
                        textFile.write(infos_str)
                        textFile.close()
                        # ----
                        pictures = [data_noisy, data_true, out, est_noise]
                        pic_paths = [img_folder+'/Img_in', img_folder+'/Img_true', img_folder+'/Img_out', img_folder+'/Img_out_net']
                        save_several_images(pictures, pic_paths, nrows=10, out_format='.png')

            if Test_mode is True:
                if i==1:
                    break
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
        
        



if __name__ == "__main__":

    var_noise = bool(opt.variable_noise)
    train_net(loss_name=opt.loss_name, pretrained=opt.pretrained, num_of_features=opt.num_of_features, num_of_layers=opt.num_of_layers, batchSize=opt.batchSize, epochs=opt.epochs)
