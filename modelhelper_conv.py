import torch
import torch.nn as nn

import math

from dualFB_conv import DFBNet_conv


def weights_init_kaiming(model):
    classname = model.__class__.__name__
    if hasattr(model, 'weight') and isinstance(model.weight, torch.Tensor):
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(model.weight, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(model.weight, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            model.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            if model.bias is not None:
                nn.init.constant_(model.bias.data, 0.0)


def get_model(architecture, num_of_layers=20):

    if architecture == 'DFBNet':
        net = DFBNet_conv(num_of_layers)
        lr = 1e-3
        clip_val = 1
        net_name = 'dfbnet_simple_lay_'+str(num_of_layers)
    

    net.apply(weights_init_kaiming)

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
