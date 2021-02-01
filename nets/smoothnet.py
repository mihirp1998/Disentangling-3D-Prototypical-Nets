import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.ops as ops
from archs.neural_modules import ResNet3D
import ipdb
st = ipdb.set_trace 
import utils_basic
import utils_geom
import utils_misc
import hyperparams as hyp

class SmoothNet(nn.Module):
    def __init__(self):
        print('SmoothNet...')
        super(SmoothNet, self).__init__()
        self.net = nn.Sequential(
            torch.nn.Conv3d(in_channels=hyp.feat_dim, out_channels=hyp.feat_dim, kernel_size=3, stride=1, padding=1).cuda()
        )

    def forward(self, emb3d, boxes, scores):
        out = self.net(emb3d)
        loss = 0
        if not hyp.smoothness_with_noloss:
            loss += hyp.smoothness_recons_loss_weight * torch.mean((out-emb3d)**2)
            loss += hyp.smoothness_gradient_loss_weight * self.gradient3DForBboxFace(out, boxes, scores)
        return out, loss

    
    '''
    ipdb> x.shape
    torch.Size([2, 32, 72, 72, 72])
    ipdb> bbox.shape
    torch.Size([2, 2, 2, 3])
    ipdb> scores.shape
    (2, 2)
    '''
    def gradient3DForBboxFace(self, emb3D_scenes, bbox, scores):
        # emb3D_scenes should be B x C x D x H x W
        dz_batch, dy_batch, dx_batch = utils_basic.gradient3D(emb3D_scenes, absolute=False, square=False)

        bbox = torch.clamp(bbox,min=0)
        sizes_val = [hyp.Z2-1, hyp.Y2-1, hyp.X2-1]

        gs_loss_list = []   # gradient smoothness loss
    
        for index_batch,emb_scene in enumerate(emb3D_scenes):
            gsloss = 0 
            dz, dy, dx = dz_batch[index_batch:index_batch+1], dy_batch[index_batch:index_batch+1], dx_batch[index_batch:index_batch+1]
            for index_box,box in enumerate(bbox[index_batch]):
                if scores[index_batch][index_box] > 0:
                    
                    lower,upper = torch.unbind(box)
                    lower = [torch.floor(i).to(torch.int32) for i in lower]
                    upper = [torch.ceil(i).to(torch.int32) for i in upper]
                    xmin,ymin,zmin = [max(i,0) for i in lower]

                    xmax,ymax,zmax = [min(i,sizes_val[index]) for index,i in enumerate(upper)]

                    #zmin face
                    gsloss += self.get_gradient_loss_on_bbox_surface(dz, zmin, zmin+1, ymin, ymax, xmin, xmax)
                    if zmin < sizes_val[0]:
                        gsloss += self.get_gradient_loss_on_bbox_surface(dz, zmin+1, zmin+2, ymin, ymax, xmin, xmax)
                    
                    #zmax face
                    gsloss += self.get_gradient_loss_on_bbox_surface(dz, zmax, zmax+1, ymin, ymax, xmin, xmax)
                    if zmax < sizes_val[0]:
                        gsloss += self.get_gradient_loss_on_bbox_surface(dz, zmax+1, zmax+2, ymin, ymax, xmin, xmax)
                    
                    #ymin face
                    gsloss += self.get_gradient_loss_on_bbox_surface(dy, zmin, zmax, ymin, ymin+1, xmin, xmax)
                    if ymin < sizes_val[1]:
                        gsloss += self.get_gradient_loss_on_bbox_surface(dy, zmin, zmax, ymin+1, ymin+2, xmin, xmax)
                    
                    #ymax face
                    gsloss += self.get_gradient_loss_on_bbox_surface(dy, zmin, zmax, ymax, ymax+1, xmin, xmax)
                    if ymax < sizes_val[1]:
                        gsloss += self.get_gradient_loss_on_bbox_surface(dy, zmin, zmax, ymax+1, ymax+2, xmin, xmax)
                    

                    #xmin face
                    gsloss += self.get_gradient_loss_on_bbox_surface(dx, zmin, zmax, ymin, ymax, xmin, xmin+1)
                    if xmin < sizes_val[2]:
                        gsloss += self.get_gradient_loss_on_bbox_surface(dx, zmin, zmax, ymin, ymax, xmin+1, xmin+2)
                    
                    #xmax face
                    gsloss += self.get_gradient_loss_on_bbox_surface(dx, zmin, zmax, ymin, ymax, xmax, xmax+1)
                    if xmax < sizes_val[2]:
                        gsloss += self.get_gradient_loss_on_bbox_surface(dx, zmin, zmax, ymin, ymax, xmax+1, xmax+2)
                    
            gs_loss_list.append(gsloss)

        gsloss = torch.mean(torch.tensor(gs_loss_list))
        return gsloss
    
    def get_gradient_loss_on_bbox_surface(self, emb3d, z1, z2, y1, y2, x1, x2, absolute=True, square=False):
        val = emb3d[:, :, z1:z2, y1:y2, x1:x2]
        if absolute:
            val = torch.abs(val)
        if square:
            val = val ** 2

        return torch.sum(val)