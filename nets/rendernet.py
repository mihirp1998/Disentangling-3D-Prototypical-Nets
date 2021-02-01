import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D2D as encoder3D2D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc

class RenderNet(nn.Module):
    def __init__(self):
        super(RenderNet, self).__init__()

        print('RenderNet...')
        
        self.rgb_layer = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=3, kernel_size=1, stride=1, padding=0).cuda()

    def accu_render(self, feat, occ):
        B, C, D, H, W = list(feat.shape)
        output = torch.zeros(B, C, H, W).cuda()
        alpha = torch.zeros(B, 1, H, W).cuda()
        for d in range(D):
            contrib = (alpha + occ[:,:,d,:,:]).clamp(max=1.) - alpha
            output += contrib*feat[:,:,d,:,:]
            alpha += contrib
        return output

    def forward(self, feat, occ, rgb_g, valid, summ_writer):
        total_loss = torch.tensor(0.0).cuda()
        
        rgb_feat = self.rgb_layer(feat)
        rgb_e = self.accu_render(rgb_feat, occ)
        emb_e = self.accu_render(feat, occ)
        
        # postproc
        emb_e = l2_normalize(emb_e, dim=1)
        rgb_e = torch.nn.functional.tanh(rgb_e)*0.5
        
        loss_im = l1_on_axis(rgb_e-rgb_g, 1, keepdim=True)
        summ_writer.summ_oned('render/rgb_loss', loss_im)
        summ_writer.summ_occs('render/occ', occ.unsqueeze(0), reduce_axes=[2])
        summ_writer.summ_occs('render/occ', occ.unsqueeze(0), reduce_axes=[3])
        summ_writer.summ_occs('render/occ', occ.unsqueeze(0), reduce_axes=[4])
        rgb_loss = utils_basic.reduce_masked_mean(loss_im, valid)

        total_loss = utils_misc.add_loss('render/rgb_l1_loss', total_loss, rgb_loss, hyp.render_l1_coeff, summ_writer)

        # vis
        summ_writer.summ_rgbs('render/rgb', [rgb_e, rgb_g])

        return total_loss, rgb_e, emb_e

