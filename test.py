import torch
import torch.nn as nn
import torch.nn.functional as F
from Jacob_lips import JacobianReg_l2

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def H_star_H(x):
    y = torch.zeros(x.shape).to(torch.device('cuda'))
    if len(x.shape) != 1:
        y[:,[0]] = x[:,[0]]
    return y





class DFBNet(nn.Module):
    """Define unfolded network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB -- default 3)
        num_of_layers (int): number of layers/iteration (default 20)
    """
    def __init__(self):
        super(DFBNet, self).__init__()

        self.L = nn.Linear(2, 2, bias=False).to(torch.device('cuda'))
        self.L.weight.data = torch.eye(2).to(torch.device('cuda'))
        self.L.weight.data[0,0] = 3
    def forward(self, y):
        '''
        Network definition -- training mode
        Args:
            y: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            x_ref: measurement
            l: threshold parameter
            u: dual variable
        '''
        return H_star_H(self.L(y))


x_in = torch.randn((1,2),requires_grad=True).to(torch.device('cuda'))
model = DFBNet()
print(model.L.weight)
x_out = model(x_in)
print(x_in, x_out)
print(JacobianReg_l2().Lips(x_in,x_out))