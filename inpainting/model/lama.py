import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
ours_root = os.path.abspath(os.path.join('.'))
sys.path.insert(0, ours_root)
# import ipdb;ipdb.set_trace()
from inpainting.model.networks.base_network import BaseNetwork
# from inpainting.model.networks.generator import SPADEGenerator, FF_SPADEGenerator
from inpainting.saicinpainting.training.modules.ffc import FFCResNetGenerator
from inpainting.model.networks.discriminator import MultiscaleDiscriminator

class InpaintGenerator(BaseNetwork):
    def __init__(self, opt, init_weights=True):
        super(InpaintGenerator, self).__init__()
        self.opt = opt
        input_nc = 1+3+3+1 if opt.use_mae else 1+3
        # input_nc = 1+3+3 if opt.use_mae else 1+3
        self.generator = FFCResNetGenerator(input_nc=input_nc, output_nc=opt.output_nc)

        if init_weights:
            self.init_weights()

    def forward(self, masked_img, mae_prediction, mask, mae_flag=None):
        if self.opt.use_mae:
            input = torch.cat([masked_img, mask, mae_prediction, mae_flag], 1)
            # input = torch.cat([masked_img, mask, mae_prediction], 1)
        else:
            input = torch.cat([masked_img, mask], 1)
        pred = self.generator(input)
        return pred

class Discriminator(BaseNetwork):
    def __init__(self, opt, use_sigmoid=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.discriminator = MultiscaleDiscriminator(opt)
        self.use_sigmoid = use_sigmoid

        if init_weights:
            self.init_weights()
    
    def forward(self, x):
        pred = self.discriminator(x)
        if self.use_sigmoid:
            pred = torch.sigmoid(pred)
        return pred

if __name__=='__main__':
    netG = FFCResNetGenerator(input_nc=8, output_nc=3)
    input = torch.ones(2,8, 512, 512)
    pred = netG(input)
    for h in range(512, 512+32):
        input = torch.ones(2,8,h,h)
        pred = netG(input)
        print(h , pred.shape[2])
    import ipdb;ipdb.set_trace()
    print(pred.shape)