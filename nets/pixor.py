import torch
import ipdb
st = ipdb.set_trace
import torch.nn as nn
import torch.nn.functional as Func
import math
import hyperparams as hyp
import ipdb
import torchvision.ops as ops
import time
import utils_vox
import utils_geom
import time

st = ipdb.set_trace

class counter:
    # val = torch.zeros(9).cuda()
    mean = torch.zeros(9).cuda()
    std = torch.zeros(9).cuda()
    # target_mean = torch.tensor([1.  ,  0.  , -0.02, -0.04, -0.04,  1.32,  1.25,  1.42]).cuda()
    # target_std_dev = torch.tensor([1.  , 1.  , 1.25, 1.2 , 1.  , 1.36, 1.32, 1.47]).cuda()     
    target_mean = torch.tensor([-0.25,  0.01,  0.26,  0.46,  0.26,  0.62,  0.69,  1.46]).cuda()
    target_std_dev = torch.tensor([0.93, 0.26, 0.53, 0.89, 1.11, 0.13, 0.16, 0.19]).cuda() 
def compute_std_sum(datapoints):
    counter.std += torch.sum((counter.mean.unsqueeze(0) - datapoints)**2, dim=0)

def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=bias)
    return nn.Conv3d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def maskFOV_on_BEV(shape, fov=88.0):

    height = shape[0]
    width = shape[1]


    fov = fov / 2

    x = np.arange(width)
    y = np.arange(-height//2, height//2)

    xx, yy = np.meshgrid(x, y)
    angle = np.arctan2(yy, xx) * 180 / np.pi

    in_fov = np.abs(angle) < fov
    in_fov = torch.from_numpy(in_fov.astype(np.float32))

    return in_fov
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, use_bn=True):
        super(Bottleneck, self).__init__()
        bias = not use_bn
        self.use_bn = use_bn
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(residual + out)
        return out


class BackBone(nn.Module):

    def __init__(self, input_channels, block, num_block, geom, use_bn=True):
        super(BackBone, self).__init__()

        self.use_bn = use_bn

        # Block 1
        self.conv1 = conv3x3x3(input_channels, 32)
        self.conv2 = conv3x3x3(32, 32)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        # Block 2-5
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])
        # Lateral layers
        self.latlayer1 = nn.Conv3d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(192, 96, kernel_size=1, stride=1, padding=0)
        # self.latlayer4 = nn.Conv3d(96, 48, kernel_size=1, stride=1, padding=0)
        # Top-down layers
        self.deconv1 = nn.ConvTranspose3d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv3 = nn.ConvTranspose3d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        # p = 0 if geom['label_shape'][1] == 175 else 1

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)
        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)
        # l2 = self.latlayer4(c2)
        # p3 = l2 + self.deconv3(p4)
        # return p3
        return p4

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv3d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y



class Header(nn.Module):

    def __init__(self,use_bn=True,input_channels=32):
        super(Header, self).__init__()
        num_channels = 32
        self.use_bn = use_bn
        bias = not use_bn
        # st()
        self.conv1 = conv3x3x3(input_channels, num_channels, bias=bias)
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.conv2 = conv3x3x3(num_channels, num_channels, bias=bias)
        self.bn2 = nn.BatchNorm3d(num_channels)
        self.conv3 = conv3x3x3(num_channels, num_channels, bias=bias)
        self.bn3 = nn.BatchNorm3d(num_channels)
        self.conv4 = conv3x3x3(num_channels, num_channels, bias=bias)
        self.bn4 = nn.BatchNorm3d(num_channels)
        self.clshead = conv3x3x3(num_channels, 1, bias=True)
        self.reghead = conv3x3x3(num_channels, 8, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        cls_head = self.clshead(x)
        # st()
        cls = torch.sigmoid(cls_head)
        reg = self.reghead(x)
        return cls, reg

class Decoder(nn.Module):

    def __init__(self, geom):
        super(Decoder, self).__init__()
        # make sure this is always maintained as width, length and height of the car
        # considering camR is pointing in the direction of the road:
        # width, height and length of the car are x,y and z
        # sin_t and cos_t are wrt to the y axis
        self.geometry = [geom["W1"], geom["W2"], geom["H1"], geom["H2"], geom["L1"], geom["L2"]]
        # self.grid_size = 1.0

    def forward(self, points,label_shape,scores=None):
        _,_,Z,Y,X = list(points.shape)
        z_grid_size = (self.geometry[5] - self.geometry[4])/float(Z)
        y_grid_size = (self.geometry[3] - self.geometry[2])/float(Y)
        x_grid_size = (self.geometry[1] - self.geometry[0])/float(X)
        device = torch.device('cuda')

        for i in range(8):
            points[:, i+1,:, :, :] = points[:, i+1,:, :, :] * counter.target_std_dev[i] + counter.target_mean[i]
        all_points_cls, all_points_reg = points.split([1, 8], dim=1)
        
        cos_t, sin_t, dx, dy, dz, log_w, log_h, log_l = torch.chunk(all_points_reg, 8, dim=1)
        

        theta = torch.atan2(sin_t, cos_t)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        x = torch.arange(self.geometry[0], self.geometry[1], x_grid_size, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[2], self.geometry[3], y_grid_size, dtype=torch.float32, device=device)
        z = torch.arange(self.geometry[4], self.geometry[5], z_grid_size, dtype=torch.float32, device=device)
        zz, yy, xx = torch.meshgrid([z, y, x])
        centre_z = zz + dz
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        h = log_h.exp()
        w = log_w.exp()
        bboxes_theta = []
        scores_theta = []
        for b in range(hyp.B):
            pixor_tgt_indexes = torch.where(all_points_cls[b:b+1]>0.5)
            if len(pixor_tgt_indexes[0]) >0:
                # init = time.time()
                centre_x_filter = centre_x[b:b+1][pixor_tgt_indexes]
                centre_y_filter = centre_y[b:b+1][pixor_tgt_indexes]
                centre_z_filter = centre_z[b:b+1][pixor_tgt_indexes]
                scores_filter = all_points_cls[b:b+1][pixor_tgt_indexes]
                l_filter = l[b:b+1][pixor_tgt_indexes]
                w_filter = w[b:b+1][pixor_tgt_indexes]
                h_filter = h[b:b+1][pixor_tgt_indexes]
                theta_filter = theta[b:b+1][pixor_tgt_indexes]
                # st()
                box_theta = torch.stack([centre_x_filter,centre_y_filter,centre_z_filter,w_filter,h_filter,l_filter,torch.zeros_like(theta_filter),theta_filter,torch.zeros_like(theta_filter)],dim=1).unsqueeze(0)
                box_corners = utils_geom.transform_boxes_to_corners(box_theta) 

                B,N,_,_ = list(box_corners.shape)
                assert B == 1

                xmin = torch.min(box_corners[...,0],dim=-1).values
                ymin = torch.min(box_corners[...,1],dim=-1).values
                zmin = torch.min(box_corners[...,2],dim=-1).values

                xmax = torch.max(box_corners[...,0],dim=-1).values
                ymax = torch.max(box_corners[...,1],dim=-1).values
                zmax = torch.max(box_corners[...,2],dim=-1).values

                xy_box = torch.stack([xmin,ymin,xmax,ymax],dim=-1).reshape([B*N,-1])
                yz_box = torch.stack([ymin,zmin,ymax,zmax],dim=-1).reshape([B*N,-1])
                # st()
                selected_bboxes_idx_xy = ops.nms(
                    xy_box,
                    scores_filter,
                    0.5)
                # selected_ones =xy_box[selected_bboxes_idx_xy]
                selected_bboxes_idx_yz = ops.nms(
                    yz_box,
                    scores_filter,
                    0.5)
                # print("decoder forward",time.time() - init)
                # st()
                selected_bboxes_idx = torch.unique(torch.cat([selected_bboxes_idx_xy, selected_bboxes_idx_yz], dim=0)) 
                selected_3d_bboxes = box_theta[:,selected_bboxes_idx] # this is (selected_bbox, 3, 2)
                scores_filter_selected = scores_filter[selected_bboxes_idx]
                B,N_filtered,_ = list(selected_3d_bboxes.shape)
                selected_3d_bboxes = Func.pad(selected_3d_bboxes,pad=[0,0,0,hyp.N-N_filtered])
                scores_filter_selected_padded =  Func.pad(scores_filter_selected,pad=[0,hyp.N-N_filtered])
                # st()
                # scores_filter_padded = Func.pad(scores_filter,pad=[0,0,0,hyp.N-N_filtered])
                bboxes_theta.append(selected_3d_bboxes)
                scores_theta.append(scores_filter_selected_padded)
            else:
                bboxes_theta.append(torch.zeros([1,hyp.N,9]).cuda())
                scores_theta.append(torch.zeros([hyp.N]).cuda())
        bboxes_theta_stacked = torch.cat(bboxes_theta,dim=0)
        scores_theta_stacked = torch.stack(scores_theta,dim=0)
        return bboxes_theta_stacked,scores_theta_stacked


class PIXOR(nn.Module):
    def __init__(self, geom, use_bn=True, decode=False):
        super(PIXOR, self).__init__()
        self.use_decode = decode
        input_dim = geom["input_shape"][-1]
        # self.cam_fov_mask = maskFOV_on_BEV(geom['label_shape'])
        # use_bn = False
        if not hyp.do_pixor_det:
            self.backbone = BackBone(input_dim,Bottleneck, [3, 6, 6, 3], geom, use_bn)
        self.header = Header(use_bn)
        self.corner_decoder = Decoder(geom)
        # st()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.header.clshead.weight.data.fill_(-math.log((1.0-prior)/prior))
        self.header.clshead.bias.data.fill_(0)
        self.header.reghead.weight.data.fill_(0)
        self.header.reghead.bias.data.fill_(0)

    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, x):
        device = torch.device('cuda')
        if x.is_cuda:
            device = x.get_device()
        if not hyp.do_pixor_det:
            features = self.backbone(x)
        else:
            features = x

        # st()
        cls, reg = self.header(features)
        # self.cam_fov_mask = self.cam_fov_mask.to(device)
        # cls = cls * self.cam_fov_mask
        # st()
        if self.use_decode:
            decoded = self.corner_decoder(reg)
            # Return tensor(Batch_size, height, width, channels)
            #decoded = decoded.permute(0, 2, 3, 1)
            #cls = cls.permute(0, 2, 3, 1)
            #reg = reg.permute(0, 2, 3, 1)
            # st()
            pred = torch.cat([cls, reg, decoded], dim=1)
        else:
            pred = torch.cat([cls, reg], dim=1)

        return pred

def bbox_coordinates_to_world_coordinates(bbox_theta_memR_i):
    xc,yc,zc,lx,ly,lz,rx,ry,rz = bbox_theta_memR_i
    xvals = torch.arange(-lx/2, lx/2+1)
    yvals = torch.arange(-ly/2, ly/2+1)
    zvals = torch.arange(-lz/2, lz/2+1)

    meshz, meshy, meshx = torch.meshgrid(zvals, yvals, xvals)
    ref_T_obj = utils_geom.convert_box_to_ref_T_obj(bbox_theta_memR_i.unsqueeze(0))
    xyz_bbox = torch.stack([meshx.reshape(-1), meshy.reshape(-1), meshz.reshape(-1)], dim=1).unsqueeze(0)
    xyz_world = utils_geom.apply_4x4(ref_T_obj.cuda(), xyz_bbox.cuda()).squeeze(0)
    x_world, y_world, z_world = xyz_world.unbind(1)
    shape = meshz.shape
    return x_world.reshape(shape), y_world.reshape(shape), z_world.reshape(shape)

def clamp_boxes(bbox_corners_memR,sizes):
    # bbox_corners_memR.shape
    st()
    return bbox_theta_memR

def get_points_in_rotated_bbox_vectorized(bbox_theta_memR, bbox_corners_memR, bbox_theta_camR, unp_visRs, summ_writer,pixorMemSize):
    A = bbox_corners_memR[:,:,0] # torch.Size([8, 8, 3]) batch X num_boxes X xyz
    pixorZ, pixorY, pixorX = pixorMemSize[:3]
    pixor_labels = torch.zeros(hyp.B, 9, pixorZ, pixorY, pixorX).cuda()
    vis_done_for_iteration = False
    for batch in range(A.shape[0]): # loop over batch
        for boxnum in range(A.shape[1]): # loop over bbox
            if not is_valid_bbox(bbox_theta_camR[batch, boxnum], bbox_corners_memR[batch, boxnum],pixorMemSize):
                continue
            bbox_theta_memR_i = bbox_theta_memR[batch, boxnum]
            # bbox_theta_memR_i = clamp_boxes(bbox_corners_memR[batch, boxnum],[pixorZ, pixorY, pixorX])
            # st()
            memRx, memRy, memRz = bbox_coordinates_to_world_coordinates(bbox_theta_memR_i)
            grid_shape = memRz.shape # should be pixorZ, pixorY, pixorX
            points_memR = torch.stack([memRx.reshape(-1), memRy.reshape(-1), memRz.reshape(-1)], dim=1).unsqueeze(0)
            points_camR = utils_vox.Mem2Ref(points_memR.float(), pixorZ, pixorY, pixorX).squeeze(0)
            camRx, camRy, camRz = points_camR.unbind(1)
            camRx, camRy, camRz = camRx.reshape(grid_shape), camRy.reshape(grid_shape), camRz.reshape(grid_shape) 
            yaw = bbox_theta_camR[batch, boxnum,7]
            # print(yaw)
            xc,yc,zc,lx,ly,lz = bbox_theta_camR[batch, boxnum, :6]
            # print(lx,ly,lz,"vectorized")

            req_target = torch.tensor([1,torch.cos(yaw), torch.sin(yaw), xc, yc, zc, torch.log(lx), torch.log(ly), torch.log(lz)]).cuda()
            memRz, memRy, memRx = memRz.long(), memRy.long(), memRx.long()
            pixor_labels[batch,  :, memRz, memRy, memRx] = req_target.view(-1,1,1,1)
            pixor_labels[batch, 3, memRz, memRy, memRx] -= camRx
            pixor_labels[batch, 4, memRz, memRy, memRx] -= camRy
            pixor_labels[batch, 5, memRz, memRy, memRx] -= camRz
            if hyp.calculate_mean:
                permuted_pixor_labels = pixor_labels.permute(0, 2, 3, 4, 1)
                counter.mean += torch.sum(permuted_pixor_labels[batch, memRz, memRy, memRx, :].view(-1, 9), dim=0)
                print("mean sum is: ", counter.mean)
            elif hyp.calculate_std:
                permuted_pixor_labels = pixor_labels.permute(0, 2, 3, 4, 1)                
                compute_std_sum(permuted_pixor_labels[batch, memRz, memRy, memRx, :].view(-1, 9))
                print("std sum is: ", counter.std)
            else:
                x_input = pixor_labels[batch, 1:9, memRz, memRy, memRx]
                x_input = (x_input -counter.target_mean.reshape([8,1,1,1]))/counter.target_std_dev.reshape([8,1,1,1])
                pixor_labels[batch, 1:9, memRz, memRy, memRx] = x_input                
            if not vis_done_for_iteration:
                vis_done_for_iteration = True
                # print("printing points")
                # st()
                unp_visRs = Func.interpolate(unp_visRs, size=[pixorMemSize[0],pixorMemSize[2]])[0]
                # st()
                summ_writer.summ_points_on_mem("pixor/box_points",unp_visRs, points_memR[0].long())
            # print(pixor_labels[batch,:, memRz, memRy, memRx],"vectorized")
    return pixor_labels

def is_valid_bbox(bbox_theta_camR, bbox_corners_memR,pixorMemSize):
    pixorZ, pixorY, pixorX = pixorMemSize[:3]
    if torch.abs(bbox_theta_camR[3]) <=1e-5 or torch.abs(bbox_theta_camR[4]) <= 1e-5 or torch.abs(bbox_theta_camR[5]) <= 1e-5:
        return False
    # x, y, z = bbox_corners_memR.unbind(1)
    # if x.min()<0 or x.max()>=pixorX-1 or y.min()<0 or y.max()>=pixorY-1 or z.min()<0 or z.max()>=pixorZ-1:
    #     return False
    return True

# defining the corners as: clockwise backcar face, clockwise frontcar face:
#   E -------- F
#  /|         /|
# A -------- B .
# | |        | |
# . H -------- G
# |/         |/
# D -------- C
# camR
# AE is Z (lenght of car)
# AD is Y (height of car)
# AB is Z (Width of car)

# the ordered eight indices are:
# A E D H B F C G

def get_points_in_rotated_bbox(bbox_corners_memR, bbox_theta_camR, unp_visRs, summ_writer, target_size):#torch.Size([8, 8, 8, 3])
    '''
    We will find points lying inside the 2d box and then expand those points along height (y) dimension.
    '''
    # xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(bbox_theta, axis=1)
    points = []
    A = bbox_corners_memR[:,:,0,[0,2]] # torch.Size([8, 8, 3]) batch X num_boxes X xyz
    E = bbox_corners_memR[:,:,1,[0,2]] # extract x and z coordinates
    B = bbox_corners_memR[:,:,4,[0,2]]
    F = bbox_corners_memR[:,:,5,[0,2]]

    ymin = bbox_corners_memR[:,:,0,1] # y for A
    ymax = bbox_corners_memR[:,:,2,1] # y for D
    pixor_labels = torch.zeros(hyp.B, 9, target_size[0], target_size[1], target_size[2]).cuda()
    corners = torch.zeros(4,2)
    from collections import defaultdict
    vis_done_for_iteration = False # Just visualize for first item of batch    
    for batch in range(A.shape[0]): # loop over batch
        for boxnum in range(A.shape[1]): # loop over bbox
            if not is_valid_bbox(bbox_theta_camR[batch, boxnum], bbox_corners_memR[batch, boxnum],target_size):
                continue
            # bbox_theta_memR_i = clamp_boxes(bbox_corners_memR[batch, boxnum],[pixorZ, pixorY, pixorX])
            # st()                
            corners[0] = A[batch, boxnum] # x,z
            corners[1] = E[batch, boxnum]
            corners[2] = B[batch, boxnum]
            corners[3] = F[batch, boxnum]

            points = get_points_in_a_rotated_box_3D(corners, ymin[batch, boxnum].long(), ymax[batch, boxnum].long(), target_size[0], target_size[2])
            points_memR = torch.tensor(points).unsqueeze(1).cuda()

            if points_memR.ndim < 3:
                continue
            # print("points_memR shape is: ", points_memR.shape)
            points_camR = utils_vox.Mem2Ref(points_memR.float(), target_size[0], target_size[1], target_size[2])
            if not vis_done_for_iteration:
                vis_done_for_iteration = True
                unp_visRs = Func.interpolate(unp_visRs, size=[target_size[0],target_size[2]])[0]
                summ_writer.summ_points_on_mem("pixor/box_points",unp_visRs, points_memR[:,0])
            for pointnum in range(len(points)):
                xx_memR = points[pointnum][0]
                yy_memR = points[pointnum][1]
                zz_memR = points[pointnum][2]
                point = points_camR[pointnum,0]
                # st()
                xx = point[0]
                yy = point[1]
                zz = point[2]
                yaw = bbox_theta_camR[batch, boxnum,7].clone()
                # st()
                # print(yaw)
                xc,yc,zc,lx,ly,lz = bbox_theta_camR[batch, boxnum, :6].clone()
                # print(lx,ly,lz,"rotated")
                req_target = [torch.cos(yaw), torch.sin(yaw), xc, yc, zc, torch.log(lx), torch.log(ly), torch.log(lz)]

                req_target[2] -= xx
                req_target[3] -= yy
                req_target[4] -= zz


                pixor_labels[batch, 0, zz_memR, yy_memR, xx_memR] = 1
                pixor_labels[batch, 1:9, zz_memR, yy_memR, xx_memR] = torch.tensor(req_target).cuda()
                # st()
                if hyp.calculate_mean:
                    counter.mean += pixor_labels[batch,:, zz_memR, yy_memR, xx_memR]
                    print("mean sum is: ", counter.mean)
                elif hyp.calculate_std:
                    compute_std_sum(pixor_labels[batch, :, zz_memR, yy_memR, xx_memR].unsqueeze(0))
                    print("std sum is: ", counter.std)
                else:
                    x_input = pixor_labels[batch, 1:9, zz_memR, yy_memR, xx_memR]
                    x_input = (x_input -counter.target_mean)/counter.target_std_dev
                    pixor_labels[batch, 1:9, zz_memR, yy_memR, xx_memR] = x_input
    return pixor_labels


def find_centres(pixor_labels,gt_boxes):
    pixor_cls,pixor_reg = pixor_labels.split([1, 8], dim=1)
    pixor_tgt_indexes = torch.where(pixor_cls==1.0)
    pixor_reg_filtered = pixor_reg[pixor_tgt_indexes[0],:,pixor_tgt_indexes[2],pixor_tgt_indexes[3],pixor_tgt_indexes[4]]

def minZ(x0, z0, x1, z1, x):
    if x0 == x1:
        # vertical line, z0 is lowest
        return int(math.floor(z0))

    m = (z1 - z0) / (x1 - x0)

    if m >= 0.0:
        # lowest point is at left edge of pixel column
        return int(math.floor(z0 + m * (x - x0)))
    else:
        # lowest point is at right edge of pixel column
        return int(math.floor(z0 + m * ((x + 1.0) - x0)))


def maxZ(x0, z0, x1, z1, x):
    if x0 == x1:
        # vertical line, z1 is highest
        return int(math.ceil(z1))

    m = (z1 - z0) / (x1 - x0)

    if m >= 0.0:
        # highest point is at right edge of pixel column
        return int(math.ceil(z0 + m * ((x + 1.0) - x0)))
    else:
        # highest point is at left edge of pixel column
        return int(math.ceil(z0 + m * (x - x0)))
def get_points_in_a_rotated_box_3D(corners, ymin, ymax,zmax, xmax):
    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, lz = l
    rx, rz = r
    bx, bz = b
    tx, tz = t
    m1x, m1z = m1
    m2x, m2z = m2

    xmin = 0
    zmin = 0

    # xmax = hyp.pixorX
    # zmax = hyp.pixorY

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,zmin)-(xmax,zmax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    bzi = max(int(math.ceil(bz)), zmin)
    tzi = min(int(math.floor(tz)), zmax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            z1 = minZ(lx, lz, bx, bz, xf)
            z2 = maxZ(lx, lz, tx, tz, xf)
        elif xf < m2x:
            if m1z < m2z:
                # Phase IIa: left/bottom --> top/right
                z1 = minZ(bx, bz, rx, rz, xf)
                z2 = maxZ(lx, lz, tx, tz, xf)
            else:
                # Phase IIb: left/top --> bottom/right
                z1 = minZ(lx, lz, bx, bz, xf)
                z2 = maxZ(tx, tz, rx, rz, xf)
        else:
            # Phase III: bottom/top --> right
            z1 = minZ(bx, bz, rx, rz, xf)
            z2 = maxZ(tx, tz, rx, rz, xf)

        z1 = max(z1, bzi)
        z2 = min(z2, tzi)

        for z in range(z1, z2):
            for y in range(ymin,ymax+1):
                pixels.append((x, y, z))
    return pixels


def get_pixor_regression_targets(unp_visRs,summ_writer,pixorMemSize,gt_boxes):
    boxlist_camRs_theta = gt_boxes
    pixorZ, pixorY, pixorX = pixorMemSize[:3]
    boxlist_pixor_memRs_theta = utils_vox.convert_boxlist_camR_to_memR(boxlist_camRs_theta, pixorZ, pixorY, pixorX)
    boxlist_pixor_memRs_corners = utils_geom.transform_boxes_to_corners(boxlist_pixor_memRs_theta)
    start_time = time.time()
    pixor_labels = get_points_in_rotated_bbox_vectorized(boxlist_pixor_memRs_theta,boxlist_pixor_memRs_corners, boxlist_camRs_theta, unp_visRs, summ_writer,pixorMemSize)

    # pixor_labels = get_points_in_rotated_bbox(boxlist_pixor_memRs_corners, boxlist_camRs_theta, unp_visRs, summ_writer,pixorMemSize)


    # indexes = torch.where(pixor_labels[0,0]==1.0)
    # indexes_old = torch.where(pixor_labels_old[0,0]==1.0)


    # val = pixor_labels[0,6,indexes[0],indexes[1],indexes[2]]
    # val_old = pixor_labels_old[0,6,indexes_old[0],indexes_old[1],indexes_old[2]]
    # print(pixor_labels_old[0,3:6,indexes_old[0],indexes_old[1],indexes_old[2]])
    # print(pixor_labels[0,3:6,indexes[0],indexes[1],indexes[2]])
    # st()
    return pixor_labels
# tensor(4.5625, device='cuda:0') tensor(3.5000, device='cuda:0') tensor(4.5625, device='cuda:0') vectorized
# MoviePy - Building file /tmp/tmp_76f5ret.gif with imageio.
# tensor(3.5000, device='cuda:0') tensor(3.5000, device='cuda:0') tensor(3.5000, device='cuda:0') vectorized    

# tensor(3.4375, device='cuda:0') tensor(3.5000, device='cuda:0') tensor(3.4375, device='cuda:0') vectorized
# MoviePy - Building file /tmp/tmpiv732etg.gif with imageio.
# tensor(8.7500, device='cuda:0') tensor(1.1875, device='cuda:0') tensor(3.1250, device='cuda:0') vectorized 
def test_decoder(decode = True):
    geom = {
        "L1": -12.0,
        "L2": 12.0,
        "W1": -12.0,
        "W2": 12.0,
        "H1": -12.0,
        "H2": 12.0,
        "input_shape": [96, 96, 96, 1],
        "label_shape": [24, 24, 24, 7]
    }
    use_bn = False
    pixor = PIXOR(geom)
    pixor.cuda()
    zero_val = torch.zeros([1,1,96, 96, 96]).cuda()
    pred_val = pixor(zero_val)
    # st()
    print("Testing PIXOR decoder")

if __name__ == "__main__":
    test_decoder()
