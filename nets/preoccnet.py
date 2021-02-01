import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder3D
# import archs.sparse_invar_encoder3D
import hyperparams as hyp
from utils_basic import *
import utils_improc
import utils_misc

class PreoccNet(nn.Module):
    def __init__(self):
        super(PreoccNet, self).__init__()

        print('PreoccNet...')

        self.net = nn.Sequential(
            archs.encoder3D.Net3D(in_channel=5, pred_dim=1, chans=32),
        ).cuda()

    def forward(self, feat, occ_g, free_g, summ_writer):
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = list(feat.shape)
        # feat is B x C x Z x Y x X
        # occ_g is B x 1 x Z x Y x X

        if hyp.preocc_do_flip:
            # randomly flip the input
            flip0 = torch.rand(1)
            flip1 = torch.rand(1)
            flip2 = torch.rand(1)
            if flip0 > 0.5:
                # transpose width/depth (rotate 90deg)
                feat = feat.permute(0,1,4,3,2)
            if flip1 > 0.5:
                # flip depth
                feat = feat.flip(2)
            if flip2 > 0.5:
                # flip width
                feat = feat.flip(4)
        
        feat = self.net(feat)

        if hyp.preocc_do_flip:
            if flip2 > 0.5:
                # unflip width
                feat = feat.flip(4)
            if flip1 > 0.5:
                # unflip depth
                feat = feat.flip(2)
            if flip0 > 0.5:
                # untranspose width/depth
                feat = feat.permute(0,1,4,3,2)
        
        # this is half res, so let's bring it up
        occ_e_ = F.interpolate(feat, scale_factor=2)
        # occ_e_ is B x 1 x Z x Y x X

        # smooth loss
        dz, dy, dx = gradient3D(occ_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        smooth_loss = torch.mean(smooth_vox)
        summ_writer.summ_oned('preocc/smooth_loss', torch.mean(smooth_vox, dim=3))
        total_loss = utils_misc.add_loss('preocc/smooth_loss', total_loss, smooth_loss, hyp.preocc_smooth_coeff, summ_writer)
    
        occ_e = F.sigmoid(occ_e_)
        occ_e_binary = torch.round(occ_e)

        summ_writer.summ_oned('preocc/reg_loss', torch.mean(occ_e, dim=3))
        total_loss = utils_misc.add_loss('preocc/regularizer_loss', total_loss, torch.mean(occ_e), hyp.preocc_reg_coeff, summ_writer)

        # collect some accuracy stats 
        occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
        free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
        either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
        either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
        acc_occ = reduce_masked_mean(occ_match, occ_g)
        acc_free = reduce_masked_mean(free_match, free_g)
        acc_total = reduce_masked_mean(either_match, either_have)

        summ_writer.summ_scalar('preocc/acc_occ', acc_occ.cpu().item())
        summ_writer.summ_scalar('preocc/acc_free', acc_free.cpu().item())
        summ_writer.summ_scalar('preocc/acc_total', acc_total.cpu().item())
        
        amount_occ = torch.mean(occ_e_binary)
        summ_writer.summ_scalar('preocc/amount_occ', amount_occ.cpu().item())

        # vis
        summ_writer.summ_occ('preocc/occ_g', occ_g, reduce_axes=[2,3])
        summ_writer.summ_occ('preocc/free_g', free_g, reduce_axes=[2,3]) 
        summ_writer.summ_occ('preocc/occ_e', occ_e, reduce_axes=[2,3])
        summ_writer.summ_occ('preocc/occ_e_binary', occ_e_binary, reduce_axes=[2,3])
        
        prob_loss = self.compute_loss(occ_e_, occ_g, free_g, summ_writer)
        total_loss = utils_misc.add_loss('preocc/prob_loss', total_loss, prob_loss, hyp.preocc_coeff, summ_writer)

        # compute final computation mask (for later nets)

        # first fatten the gt; we will include all this
        weights = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))
        occ_g_fat = F.conv3d(occ_g, weights, padding=1)
        occ_g_fat = torch.clamp(occ_g_fat, 0, 1)
        # to save us in the case that occ_g_fat is already beyond our target density,
        # let's add some uncertainty to it, so that we have a chance to drop some of it
        # (in practice, i never see final density drop to 0, which means this is not a big risk)
        occ_g_mask = torch.FloatTensor(B, 1, Z, Y, X).uniform_(0.8, 1.0).cuda()
        occ_g_fat *= occ_g_mask
            
        summ_writer.summ_occ('preocc/occ_g_fat', occ_g_fat)

        # definitely exclude the known free voxels
        comp_mask = torch.clamp(occ_e.detach()-free_g, 0, 1)
        # definitely include the known occ voxels
        comp_mask = torch.clamp(comp_mask+occ_g_fat, 0, 1)
        
        summ_writer.summ_occ('preocc/comp_mask', comp_mask.round())

        # print('trimming comp_mask to have at most %.2f density' % hyp.preocc_density_coeff)
        comp_mask[comp_mask < 0.5] = 0.0
        if hyp.preocc_density_coeff > 0:
            while torch.mean(comp_mask.round()) > hyp.preocc_density_coeff:
                comp_vec = comp_mask.reshape(-1)
                nonzero_min = torch.min(comp_vec[comp_vec > 0])
                comp_mask[comp_mask < (nonzero_min + 0.05)] = 0.0
                # print('setting values under %.2f+0.05 to zero; now density is %.2f' % (
                #     nonzero_min.cpu().numpy(), torch.mean(comp_mask.round()).cpu().numpy()))
        comp_mask = torch.round(comp_mask)
        summ_writer.summ_occ('preocc/comp_mask_trimmed', comp_mask)

        amount_comp = torch.mean(comp_mask)
        summ_writer.summ_scalar('preocc/amount_comp', amount_comp.cpu().item())
                
        return total_loss, comp_mask

    def compute_loss(self, pred, occ, free, summ_writer):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_, dim=3)
        summ_writer.summ_oned('preocc/prob_loss', loss_vis, summ_writer)

        pos_loss = reduce_masked_mean(loss, pos)
        neg_loss = reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

