import torch

import torch.nn as nn

import sys
sys.path.append("..")
import ipdb
st = ipdb.set_trace
import hyperparams as hyp
import archs.encoder3D as encoder3D
if hyp.feat_do_sb:
    import archs.sparse_encoder3D as sparse_encoder3D
if hyp.feat_do_sparse_invar:
    import archs.sparse_invar_encoder3D as sparse_invar_encoder3D
from utils_basic import *
import utils_geom
import utils_vox
import utils_misc

EPS = 1e-4
class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        if hyp.feat_do_sb:
            if hyp.feat_do_resnet:
                self.net = sparse_encoder3D.SparseResNet3D(in_channel=4, pred_dim=hyp.feat_dim).cuda()
            else:
                self.net = sparse_encoder3D.SparseNet3D(in_channel=4, pred_dim=hyp.feat_dim).cuda()
        else:
            if hyp.feat_do_resnet:
                self.net = encoder3D.ResNet3D(in_channel=4, pred_dim=hyp.feat_dim).cuda()
            elif hyp.feat_do_sparse_invar:
                # self.net = sparse_invar_encoder3D.ResNet3D(in_channel=4, pred_dim=hyp.feat_dim).cuda()
                self.net = sparse_invar_encoder3D.Custom3D(in_channel=4, pred_dim=hyp.feat_dim).cuda()
            else:
                if hyp.no_bn:
                    if hyp.imgnet:
                        if hyp.imgnet_v1:
                            self.net = encoder3D.Net3D_NOBN(in_channel=65, pred_dim=hyp.feat_dim).cuda()                        
                        else:
                            self.net = encoder3D.Net3D_NOBN(in_channel=193, pred_dim=hyp.feat_dim).cuda()                            
                    elif hyp.onlyocc:                        
                        self.net = encoder3D.Net3D_NOBN(in_channel=1, pred_dim=hyp.feat_dim).cuda()
                    else:
                        self.net = encoder3D.Net3D_NOBN(in_channel=4, pred_dim=hyp.feat_dim).cuda()
                else:
                    self.net = encoder3D.Net3D(in_channel=4, 
                                               pred_dim=hyp.feat_dim,
                                               do_quantize=hyp.feat_quantize).cuda()
        # print(self.net.named_parameters)

    def forward(self, feat, summ_writer, mask=None,prefix=""):
        total_loss = torch.tensor(0.0).cuda()
        B, C, D, H, W = list(feat.shape)
        if not hyp.onlyocc:
            summ_writer.summ_feat(f'feat/{prefix}feat0_input', feat)
        
        if hyp.feat_do_rt:
            # apply a random rt to the feat
            # Y_T_X = utils_geom.get_random_rt(B, r_amount=5.0, t_amount=8.0).cuda()
            # Y_T_X = utils_geom.get_random_rt(B, r_amount=1.0, t_amount=8.0).cuda()
            Y_T_X = utils_geom.get_random_rt(B, r_amount=1.0, t_amount=4.0).cuda()
            feat = utils_vox.apply_4x4_to_vox(Y_T_X, feat)
            summ_writer.summ_feat(f'feat/{prefix}feat1_rt', feat)

        if hyp.feat_do_flip:
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
            summ_writer.summ_feat(f'feat/{prefix}feat2_flip', feat)
        
        if hyp.feat_do_sb:
            feat = self.net(feat, mask)
        elif hyp.feat_do_sparse_invar:
            feat, mask = self.net(feat, mask)
        else:
            if hyp.feat_quantize:
                feat,feat_uq,loss,encodings,perplexity = self.net(feat)
                total_loss = utils_misc.add_loss('feat_loss',total_loss,
                                                 loss,hyp.feat_coeff,summ_writer)
                summ_writer.summ_scalar('feat/perplexity',perplexity)
                summ_writer.summ_histogram('feat/encodings',encodings)
                ## Visualizing encodings will make training very slow.
                ## Use this only for debugging.
                # feat_uq = feat_uq[:1]
                # encodings = encodings[:1]
                # B,C,D2,H2,W2 = feat_uq.shape
                # feat_uq = feat_uq.permute(0,2,3,4,1) # [B,D,H,W,C]
                # feat_uq = feat_uq.reshape(B*D2*H2*W2,C)
                # encodings = encodings.flatten() # [B*D2*H2*W2]
                # summ_writer.summ_embeddings('feat/emb_before_vqvae',feat_uq,encodings)
                del feat_uq,encodings,perplexity # Cleanup.
            else:
                feat = self.net(feat)
        feat = l2_normalize(feat, dim=1)
        summ_writer.summ_feat(f'feat/{prefix}feat3_out', feat)
        
        if hyp.feat_do_flip:
            if flip2 > 0.5:
                # unflip width
                feat = feat.flip(4)
            if flip1 > 0.5:
                # unflip depth
                feat = feat.flip(2)
            if flip0 > 0.5:
                # untranspose width/depth
                feat = feat.permute(0,1,4,3,2)
            summ_writer.summ_feat(f'feat/{prefix}feat4_unflip', feat)
                
        if hyp.feat_do_rt:
            # undo the random rt
            X_T_Y = utils_geom.safe_inverse(Y_T_X)
            feat = utils_vox.apply_4x4_to_vox(X_T_Y, feat)
            summ_writer.summ_feat(f'feat/{prefix}feat5_unrt', feat)

        # valid_mask = 1.0 - (feat==0).all(dim=1, keepdim=True).float()
        # if hyp.feat_do_sparse_invar:
        #     valid_mask = valid_mask * mask
        return feat,  total_loss


