import torch
import torch.nn as nn
import torch.nn.functional as F
# from spatial_correlation_sampler import SpatialCorrelationSampler

# import sys
# sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
import utils_basic
import utils_improc
import utils_misc
import utils_samp
import math

class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()

        print('FlowNet...')

        self.debug = False
        # self.debug = True
        
        self.heatmap_size = hyp.flow_heatmap_size
        self.compress_dim = 8
        # self.scales = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
        self.scales = [1.0]
        self.num_scales = len(self.scales)

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.heatmap_size,
            stride=1,
            padding=0,
            dilation_patch=1,
        ).cuda()
        
        self.flow_predictor = nn.Sequential(
            nn.Conv3d(in_channels=(self.heatmap_size**3), out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
        ).cuda()
        
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.smoothl1_mean = torch.nn.SmoothL1Loss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='none')
        self.mse_mean = torch.nn.MSELoss(reduction='mean')

    # def flow_predictor(self, cc):
    #     B, C, Z, Y, X = list(cc.shape)
    #     # cc is B x C x Z x Y x X
    #     assert(C==(self.heatmap_size**3))
        
    #     cc = cc.reshape(B, self.heatmap_size, self.heatmap_size, self.heatmap_size, -1)
    #     cc = cc.permute(0, 4, 1, 2, 3)
    #     cc = cc.reshape(-1, self.heatmap_size, self.heatmap_size, self.heatmap_size)
    #     mean_heat = torch.mean(cc.reshape(B, Z*Y*X, self.heatmap_size, self.heatmap_size, self.heatmap_size),
    #                            dim=1, keepdim=True)
    #     z, y, x = utils_basic.argmax3D(cc, hard=True)
    #     # now for every pixel we know its argmax location
    #     z = z - math.floor(self.heatmap_size/2)
    #     y = y - math.floor(self.heatmap_size/2)
    #     x = x - math.floor(self.heatmap_size/2)

    #     z = z.reshape(B, 1, Z, Y, X)
    #     y = y.reshape(B, 1, Z, Y, X)
    #     x = x.reshape(B, 1, Z, Y, X)

    #     # z = torch.mean(z, dim=1)
    #     # y = torch.mean(y, dim=1)
    #     # x = torch.mean(x, dim=1)
        
    #     flow = torch.cat([x,y,z], dim=1)

    #     flow = torch.mean(flow, dim=[2,3,4], keepdim=True)
    #     flow = flow.repeat(1, 1, Z, Y, X)
        
    #     return flow, mean_heat
    #     # summ_writer.summ_histogram('flowX0_g_nonzero_hist', g[torch.abs(g)>0.01])
        
    def generate_flow(self, feat0, feat1, sc):
        B, C, D, H, W = list(feat0.shape)
        utils_basic.assert_same_shape(feat0, feat1)

        if self.debug:
            print('scale = %.2f' % sc)
            print('inputs:')
            print(feat0.shape)
            print(feat1.shape)

        if not sc==1.0:
            # assert(sc==0.5 or sc==0.25) # please only use 0.25, 0.5, or 1.0 right now
            feat0 = F.interpolate(feat0, scale_factor=sc, mode='trilinear', align_corners=False)
            feat1 = F.interpolate(feat1, scale_factor=sc, mode='trilinear', align_corners=False)
            D, H, W = int(D*sc), int(H*sc), int(W*sc)
            if self.debug:
                print('downsamps:')
                print(feat0.shape)
                print(feat1.shape)

        feat0 = feat0.contiguous()
        feat1 = feat1.contiguous()

        cc = self.correlation_sampler(feat0, feat1)
        if self.debug:
            print('cc:')
            print(cc.shape)
        cc = cc.view(B, self.heatmap_size**3, D, H, W)

        cc = F.relu(cc) # relu works better than leaky relu here
        if self.debug:
            print(cc.shape)
        cc = utils_basic.l2_normalize(cc, dim=1)

        # flow, heat = self.flow_predictor(cc)
        flow = self.flow_predictor(cc)
        if self.debug:
            print('flow:')
            print(flow.shape)

        if not sc==1.0:
            # note 1px here means 1px/sc at the real scale
            # first let's put the pixels in the right places
            flow = F.interpolate(flow, scale_factor=(1./sc), mode='trilinear', align_corners=False)
            # now let's correct the scale
            flow = flow/sc

        if self.debug:
            print('flow up:')
            print(flow.shape)
            
        # return flow#, feat0, feat1
        return flow#, heat

    def forward(self, feat0, feat1, flow_g, mask_g, is_synth, summ_writer):
        total_loss = torch.tensor(0.0).cuda()

        B, C, D, H, W = list(feat0.shape)
        utils_basic.assert_same_shape(feat0, feat1)

        # feats = torch.cat([feat0, feat1], dim=0)
        # feats = self.compressor(feats)
        # feats = utils_basic.l2_normalize(feats, dim=1)
        # feat0, feat1 = feats[:B], feats[B:]

        flow_total_forw = torch.zeros_like(flow_g)
        flow_total_back = torch.zeros_like(flow_g)

        feat0_aligned = feat0.clone()
        feat1_aligned = feat1.clone()


        # cycle_losses = []
        # l1_losses = []

        # torch does not like it when we overwrite, so let's pre-allocate
        l1_loss_cumu = torch.tensor(0.0).cuda()
        l2_loss_cumu = torch.tensor(0.0).cuda()
        warp_loss_cumu = torch.tensor(0.0).cuda()

        summ_writer.summ_feats('flow/feats_aligned_%.2f' % 0.0, [feat0, feat1_aligned])
        feat_diff = torch.mean(utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True))
        utils_misc.add_loss('flow/feat_align_diff_%.2f' % 0.0, 0, feat_diff, 0, summ_writer)

        # print('feat0, feat1_aligned, mask')
        # print(feat0.shape)
        # print(feat1_aligned.shape)
        # print(mask_g.shape)

        hinge_loss_vox = utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True)
        hinge_loss_vox = F.relu(0.2 - hinge_loss_vox)
        summ_writer.summ_oned('flow/hinge_loss', torch.mean(hinge_loss_vox, dim=3))
        hinge_mask_vox = (torch.sum(torch.abs(flow_g), dim=1, keepdim=True) > 1.0).float()
        summ_writer.summ_oned('flow/hinge_mask', torch.mean(hinge_mask_vox, dim=3), norm=False)
        # hinge_loss = torch.mean(hinge_loss_vox)
        hinge_loss = utils_basic.reduce_masked_mean(hinge_loss_vox, hinge_mask_vox)
        total_loss = utils_misc.add_loss('flow/hinge', total_loss, hinge_loss, hyp.flow_hinge_coeff, summ_writer)
        
        for sc in self.scales:

            # flow_forw, new_feat0, new_feat1 = self.generate_flow(feat0, feat1_aligned, sc)
            # flow_back, new_feat1, new_feat1 = self.generate_flow(feat1, feat0_aligned, sc)
            # flow_forw, heat = self.generate_flow(feat0, feat1_aligned, sc)
            
            flow_forw = self.generate_flow(feat0, feat1_aligned, sc)
            flow_back = self.generate_flow(feat1, feat0_aligned, sc)

            flow_total_forw = flow_total_forw + flow_forw
            flow_total_back = flow_total_back + flow_back

            # compositional LK: warp the original thing using the cumulative flow
            feat1_aligned = utils_samp.backwarp_using_3D_flow(feat1, flow_total_forw)
            feat0_aligned = utils_samp.backwarp_using_3D_flow(feat0, flow_total_back)
            valid1_region = utils_samp.backwarp_using_3D_flow(torch.ones_like(feat1[:,0:1]), flow_total_forw)
            valid0_region = utils_samp.backwarp_using_3D_flow(torch.ones_like(feat0[:,0:1]), flow_total_forw)

            summ_writer.summ_feats('flow/feats_aligned_%.2f' % sc, [feat0, feat1_aligned],
                                   valids=[valid0_region, valid1_region])
            # summ_writer.summ_oned('flow/mean_heat_%.2f' % sc, torch.mean(heat, dim=3))
            # feat_diff = torch.mean(utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True))
            feat_diff = utils_basic.reduce_masked_mean(utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True), valid1_region*valid0_region)
            utils_misc.add_loss('flow/feat_align_diff_%.2f' % sc, 0, feat_diff, 0, summ_writer)


            if sc==1.0:
                
                warp_loss_cumu = warp_loss_cumu + feat_diff*sc
                
                l1_diff_3chan = self.smoothl1(flow_total_forw, flow_g)
                l1_diff = torch.mean(l1_diff_3chan, dim=1, keepdim=True)
                l2_diff_3chan = self.mse(flow_total_forw, flow_g)
                l2_diff = torch.mean(l2_diff_3chan, dim=1, keepdim=True)

                nonzero_mask = ((torch.sum(torch.abs(flow_g), axis=1, keepdim=True) > 0.01).float())*mask_g
                yeszero_mask = (1.0-nonzero_mask)*mask_g
                l1_loss_nonzero = utils_basic.reduce_masked_mean(l1_diff, nonzero_mask)
                l1_loss_yeszero = utils_basic.reduce_masked_mean(l1_diff, yeszero_mask)
                l1_loss_balanced = (l1_loss_nonzero + l1_loss_yeszero)*0.5
                l2_loss_nonzero = utils_basic.reduce_masked_mean(l2_diff, nonzero_mask)
                l2_loss_yeszero = utils_basic.reduce_masked_mean(l2_diff, yeszero_mask)
                l2_loss_balanced = (l2_loss_nonzero + l2_loss_yeszero)*0.5
                # l1_loss_cumu = l1_loss_cumu + l1_loss_balanced*sc
                
                l1_loss_cumu = l1_loss_cumu + l1_loss_balanced*sc
                l2_loss_cumu = l2_loss_cumu + l2_loss_balanced*sc
                
                # warp flow
                flow_back_aligned_to_forw = utils_samp.backwarp_using_3D_flow(flow_total_back, flow_total_forw.detach())
                flow_forw_aligned_to_back = utils_samp.backwarp_using_3D_flow(flow_total_forw, flow_total_back.detach())

                cancelled_flow_forw = flow_total_forw + flow_back_aligned_to_forw
                cancelled_flow_back = flow_total_back + flow_forw_aligned_to_back

                cycle_forw = self.smoothl1_mean(cancelled_flow_forw, torch.zeros_like(cancelled_flow_forw))
                cycle_back = self.smoothl1_mean(cancelled_flow_back, torch.zeros_like(cancelled_flow_back))
                cycle_loss = cycle_forw + cycle_back
                total_loss = utils_misc.add_loss('flow/cycle_loss', total_loss, cycle_loss, hyp.flow_cycle_coeff, summ_writer)

                summ_writer.summ_3D_flow('flow/flow_e_forw_%.2f' % sc, flow_total_forw*mask_g, clip=0.0)
                summ_writer.summ_3D_flow('flow/flow_e_back_%.2f' % sc, flow_total_back, clip=0.0)
                summ_writer.summ_3D_flow('flow/flow_g_%.2f' % sc, flow_g, clip=0.0)
                
                utils_misc.add_loss('flow/l1_loss_nonzero', 0, l1_loss_nonzero, 0, summ_writer)
                utils_misc.add_loss('flow/l1_loss_yeszero', 0, l1_loss_yeszero, 0, summ_writer)
                utils_misc.add_loss('flow/l1_loss_balanced', 0, l1_loss_balanced, 0, summ_writer)
                utils_misc.add_loss('flow/l2_loss_balanced', 0, l2_loss_balanced, 0, summ_writer)
                
                # total_loss = utils_misc.add_loss('flow/l1_loss_balanced', total_loss, l1_loss_balanced, hyp.flow_l1_coeff, summ_writer)
                # total_loss = utils_misc.add_loss('flow/l1_loss_balanced', total_loss, l1_loss_balanced, hyp.flow_l1_coeff, summ_writer)
                # total_loss = utils_misc.add_loss('flow/l1_loss', total_loss, l1_loss, hyp.flow_l1_coeff*(sc==1.0), summ_writer)

        if is_synth:
            total_loss = utils_misc.add_loss('flow/synth_l1_cumu', total_loss, l1_loss_cumu, hyp.flow_synth_l1_coeff, summ_writer)
            total_loss = utils_misc.add_loss('flow/synth_l2_cumu', total_loss, l2_loss_cumu, hyp.flow_synth_l2_coeff, summ_writer)
        else:
            total_loss = utils_misc.add_loss('flow/l1_cumu', total_loss, l1_loss_cumu, hyp.flow_l1_coeff, summ_writer)
            total_loss = utils_misc.add_loss('flow/l2_cumu', total_loss, l2_loss_cumu, hyp.flow_l2_coeff, summ_writer)

        # total_loss = utils_misc.add_loss('flow/warp', total_loss, feat_diff, hyp.flow_warp_coeff, summ_writer)
        total_loss = utils_misc.add_loss('flow/warp_cumu', total_loss, warp_loss_cumu, hyp.flow_warp_coeff, summ_writer)


        # feat1_aligned = utils_samp.backwarp_using_3D_flow(feat1, flow_g)
        # valid_region = utils_samp.backwarp_using_3D_flow(torch.ones_like(feat1[:,0:1]), flow_g)
        # summ_writer.summ_feats('flow/feats_aligned_g', [feat0, feat1_aligned],
        #                        valids=[valid_region, valid_region])
        # feat_diff = utils_basic.reduce_masked_mean(utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True), valid_region)
        # total_loss = utils_misc.add_loss('flow/warp_g', total_loss, feat_diff, hyp.flow_warp_g_coeff, summ_writer)
        # # hinge_loss_vox = F.relu(0.2 - hinge_loss_vox)
        
        
        # total_loss = utils_misc.add_loss('flow/cycle_loss', total_loss, torch.sum(torch.stack(cycle_losses)), hyp.flow_cycle_coeff, summ_writer)
        # total_loss = utils_misc.add_loss('flow/l1_loss', total_loss, torch.sum(torch.stack(l1_losses)), hyp.flow_l1_coeff, summ_writer)
        
        # smooth loss
        dx, dy, dz = utils_basic.gradient3D(flow_total_forw, absolute=True)
        smooth_vox_forw = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        dx, dy, dz = utils_basic.gradient3D(flow_total_back, absolute=True)
        smooth_vox_back = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        summ_writer.summ_oned('flow/smooth_loss_forw', torch.mean(smooth_vox_forw, dim=3))
        smooth_loss = torch.mean((smooth_vox_forw + smooth_vox_back)*0.5)
        total_loss = utils_misc.add_loss('flow/smooth_loss', total_loss, smooth_loss, hyp.flow_smooth_coeff, summ_writer)
    
        # flow_e = F.sigmoid(flow_e_)
        # flow_e_binary = torch.round(flow_e)

        # # collect some accuracy stats 
        # flow_match = flow_g*torch.eq(flow_e_binary, flow_g).float()
        # free_match = free_g*torch.eq(1.0-flow_e_binary, free_g).float()
        # either_match = torch.clamp(flow_match+free_match, 0.0, 1.0)
        # either_have = torch.clamp(flow_g+free_g, 0.0, 1.0)
        # acc_flow = reduce_masked_mean(flow_match, flow_g*valid)
        # acc_free = reduce_masked_mean(free_match, free_g*valid)
        # acc_total = reduce_masked_mean(either_match, either_have*valid)

        # summ_writer.summ_scalar('flow/acc_flow', acc_flow.cpu().item())
        # summ_writer.summ_scalar('flow/acc_free', acc_free.cpu().item())
        # summ_writer.summ_scalar('flow/acc_total', acc_total.cpu().item())

        # # vis
        # summ_writer.summ_flow('flow/flow_g', flow_g)
        # summ_writer.summ_flow('flow/free_g', free_g) 
        # summ_writer.summ_flow('flow/flow_e', flow_e)
        # summ_writer.summ_flow('flow/valid', valid)
        
        # prob_loss = self.compute_loss(flow_e_, flow_g, free_g, valid, summ_writer)
        # total_loss = utils_misc.add_loss('flow/prob_loss', total_loss, prob_loss, hyp.flow_coeff, summ_writer)

        # return total_loss, flow_e
        return total_loss, flow_total_forw

