import torch
import torch.nn as nn

import math

from dualFB import DFBNet, DFBNet_block_form

from dualFB_conv import DFBNet_conv, DFBNet_conv_


def weights_init_kaiming(seed,model, block):
    torch.manual_seed(seed) # Reproducibility
    classname = model.__class__.__name__
    #if hasattr(model, 'weight') and isinstance(model.weight, torch.Tensor):
        # if classname.find('Conv') != -1:
        #     nn.init.kaiming_normal_(model.weight, a=0, mode='fan_in')
        # elif classname.find('Linear') != -1:
        #     nn.init.kaiming_normal_(model.weight, a=0, mode='fan_in')
        # elif classname.find('BatchNorm') != -1:
        #     model.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        #     if model.bias is not None:
        #         nn.init.constant_(model.bias.data, 0.0)
    if classname.find('DFBNet_conv_') != -1:
        nn.init.kaiming_normal_(model.L.weight, a=0, mode='fan_in')
        for _ in range(1,len(model.linlist)):
            model.linlist[_].log_sigma = torch.nn.Parameter(1+torch.randn(1) * 0.01)
    elif classname.find('DFBNet_conv') != -1:
        for _ in range(1,len(model.linlist)):
            nn.init.kaiming_normal_(model.linlist[_].conv.weight, a=0, mode='fan_in')
            model.linlist[_].log_sigma = torch.nn.Parameter(1+torch.randn(1) * 0.01)
    else:
        if block:
            nn.init.kaiming_normal_(model.L.weight(), a=0, mode='fan_in')
        else:
            nn.init.kaiming_normal_(model.L.weight, a=0, mode='fan_in')
            #model.log_sigma = torch.nn.Parameter(1+torch.randn(1) * 0.01)
        # for _ in range(1,len(model.linlist)):
        #     model.linlist[_].log_sigma = torch.nn.Parameter(1+torch.randn(1) * 0.01)



def get_model(seed,architecture, N, S, conv, m, r, P_star, block, num_of_layers=20):

    if architecture == 'DFBNet':
        if conv==1:
            net = DFBNet_conv(num_of_layers=num_of_layers)
        elif conv == 2:
            net = DFBNet_conv_(num_of_layers=num_of_layers)
        else:
            if block: 
                net = DFBNet_block_form(N, S, m, r, P_star, num_of_layers)
            else:
                net = DFBNet(N, S, num_of_layers)

        lr = 1e-3
        clip_val = 1
        net_name = 'dfbnet_simple_lay_'+str(num_of_layers)
    
    weights_init_kaiming(seed,net,block)

    net = nn.DataParallel(net)
    if torch.cuda.is_available():  # Move to GPU if possible
        net = net.cuda()

    return net, net_name, clip_val, lr


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    try:
        model.module.load_state_dict(checkpoint, strict=True)
    except:
        model.module.load_state_dict(checkpoint.module.state_dict(), strict=True)

    model.eval()
    if 'dpir' in filename:
        for k, v in model.module.named_parameters():
            print(k)
            v.requires_grad = False

    return model


def load_model(pth=None, net=None):
    """    Having trouble loading a model that was trained using DataParallel and then sending it to DataParallel.
    The loading works but drastic slowdown.
    Latest option is to move to CPU, load, and then move back to Parallel GPU.
    """
    loaded_txt = "Loading " + pth + "...\n"
    print(loaded_txt)

    model = load_checkpoint(net, pth)

    return model
