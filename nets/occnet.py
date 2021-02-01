import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc

class OccNet(nn.Module):
    def __init__(self):
        super(OccNet, self).__init__()

        print('OccNet...')

        if not hyp.occ_do_cheap:
            self.conv3d = nn.Sequential(
                archs.encoder3D.Net3D(in_channel=4, pred_dim=8),
                nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0),
            ).cuda()
        else:
            self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
            print('conv3D, [in_channels={}, out_channels={}, ksize={}]'.format(hyp.feat_dim, 1, 1))

    def forward(self, feat, occ_g, free_g, valid, summ_writer,prefix="", log_summ = True,only_pred = False):
        total_loss = torch.tensor(0.0).cuda()

        occ_e_ = self.conv3d(feat)
        # occ_e_ is B x 1 x Z x Y x X

        # smooth loss
        dz, dy, dx = gradient3D(occ_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        
        summ_writer.summ_oned(f'occ/{prefix}smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)

        total_loss = utils_misc.add_loss(f'occ/{prefix}smooth_loss', total_loss, smooth_loss, hyp.occ_smooth_coeff, summ_writer)
    
        occ_e = F.sigmoid(occ_e_)
        occ_e_binary = torch.round(occ_e)

        # collect some accuracy stats 
        occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
        free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
        either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
        either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
        acc_occ = reduce_masked_mean(occ_match, occ_g*valid)
        acc_free = reduce_masked_mean(free_match, free_g*valid)
        acc_total = reduce_masked_mean(either_match, either_have*valid)

        if log_summ:
            summ_writer.summ_scalar(f'occ/{prefix}acc_occ', acc_occ.cpu().item())
            summ_writer.summ_scalar(f'occ/{prefix}acc_free', acc_free.cpu().item())
            summ_writer.summ_scalar(f'occ/{prefix}acc_total', acc_total.cpu().item())

            # vis
            summ_writer.summ_occ(f'occ/{prefix}occ_g', occ_g)
            summ_writer.summ_occ(f'occ/{prefix}free_g', free_g) 
            summ_writer.summ_occ(f'occ/{prefix}occ_e', occ_e)
            summ_writer.summ_occ(f'occ/{prefix}valid', valid)
        
        prob_loss = self.compute_loss(occ_e_, occ_g, free_g, valid, summ_writer,prefix=prefix)
        total_loss = utils_misc.add_loss(f'occ/{prefix}prob_loss', total_loss, prob_loss, hyp.occ_coeff, summ_writer)

        return total_loss, occ_e
    
    def predict(self,feat,prefix,summ_writer):
        occ_e_ = self.conv3d(feat)            
        occ_e = F.sigmoid(occ_e_)
        occ_e = torch.round(occ_e)
        summ_writer.summ_occ(f'aug_occ/{prefix}occ_e', occ_e)
        return occ_e

    def compute_loss(self, pred, occ, free, valid, summ_writer,prefix=""):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned(f'occ/{prefix}prob_loss', loss_vis, summ_writer)

        pos_loss = reduce_masked_mean(loss, pos*valid)
        neg_loss = reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

