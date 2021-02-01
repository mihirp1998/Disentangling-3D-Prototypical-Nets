
from archs.munit_modules import MUNITTrainer2D
from archs.munit_modules_3d import AdaINGen3d, MUNITTrainer3D, MUNITTrainer3D_Simple

import torch
import torch.nn as nn
import hyperparams as hyp
import ipdb 
st = ipdb.set_trace

class MunitNet(nn.Module):
    def __init__(self):
        super(MunitNet, self).__init__()
        if hyp.do_2d_style_munit:
            self.net = MUNITTrainer2D()
        elif hyp.do_3d_style_munit:
            self.net = MUNITTrainer3D()
        else:
            assert(False) # one should be selected.
        
    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, input1, input2):
        if hyp.do_2d_style_munit:
            # Fix this code before using it.
            recons = self.net(data)
            recons_loss = self.recon_criterion(recons, data)
            return recons, recons_loss
        elif hyp.do_3d_style_munit:
            # st()
            hyperparams = self.net.hyperparameters
            # st()
            gen_loss, sudo_input_0, sudo_input_1, recon_input_0, recon_input_1, sudo_input_0_cycle, sudo_input_1_cycle, styles, contents  = self.net.gen_update_paired(input1, input2, hyperparams)
            return gen_loss, sudo_input_0, sudo_input_1, recon_input_0, recon_input_1, sudo_input_0_cycle, sudo_input_1_cycle, styles, contents





class MunitNet_Simple(nn.Module):
    def __init__(self):
        super(MunitNet_Simple, self).__init__()
        if hyp.do_2d_style_munit:
            self.net = AdaINGen()
        elif hyp.do_3d_style_munit:
            self.net = MUNITTrainer3D_Simple()
        else:
            assert(False) # one should be selected.
        
    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, input1, input2):
        if hyp.do_2d_style_munit:
            # Fix this code before using it.
            recons = self.net(data)
            recons_loss = self.recon_criterion(recons, data)
            return recons, recons_loss
        elif hyp.do_3d_style_munit:
            # st()
            hyperparams = self.net.hyperparameters
            # st()
            gen_loss, sudo_input_0, sudo_input_1, recon_input_0, recon_input_1, sudo_input_0_cycle, sudo_input_1_cycle, styles, contents, adin = self.net.gen_update_paired(input1, input2, hyperparams)

            return gen_loss, sudo_input_0, sudo_input_1, recon_input_0, recon_input_1, sudo_input_0_cycle, sudo_input_1_cycle, styles, contents, adin