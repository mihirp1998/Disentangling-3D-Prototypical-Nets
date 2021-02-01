import torch
import torch.nn as nn
import time
import torch.nn.functional as F
# import hyperparams as hyp
# from utils_basic import *

actually_standard_conv = False
# actually_standard_conv = True

class Pad(nn.Module):
    # standard batchnorm, but allow packed (x,m) inputs
    def __init__(self, amount):
        super(Pad, self).__init__()
        self.pad = nn.ConstantPad3d(amount, 0)
    def forward(self, input):
        x, m = input
        x, m = self.pad(x), self.pad(m)
        return x, m

class BatchNorm(nn.Module):
    # standard batchnorm, but allow packed (x,m) inputs
    def __init__(self, out_channels):
        super(BatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)
    def forward(self, input):
        x, m = input
        x = self.batch_norm(x)
        return x, m

class LeakyRelu(nn.Module):
    # standard leaky relu, but allow packed (x,m) inputs
    def __init__(self):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = nn.LeakyReLU(inplace=True)
    def forward(self, input):
        x, m = input
        x = self.leaky_relu(x)
        return x, m

class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SparseConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        self.pool = nn.MaxPool3d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.pool.require_grad = False
    def forward(self, input):
        x, m = input
        mc = m.expand_as(x)
        x = x * mc
        x = self.conv(x)
        weights = torch.ones_like(self.conv.weight)
        mc = F.conv3d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc
        x = x * mc
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
        m = self.pool(m)
        return x, m

class Conv(nn.Module):
    # standard conv, but allow packed (x,m) inputs
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
    def forward(self, input):
        x, m = input
        x = self.conv(x)
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1, 1).expand_as(x)
        return x, m

# class SparseConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True):
#         super(SparseConvBlock, self).__init__()
#         self.sparse_conv = SparseConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
#     def forward(self, input):
#         x, m = input
#         x, m = self.sparse_conv((x, m))
#         assert (m.size(1)==1)
#         return x, m

# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True):
#         super(ConvBlock, self).__init__()
#         self.conv = Conv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
#     def forward(self, input):
#         x, m = input
#         x = self.conv(x)
#         assert (m.size(1)==1)
#         return x, m
    
class Net3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(Net3D, self).__init__()
        conv3d = []
        up_bn = [] #batch norm layer for deconvolution
        conv3d_transpose = []

        # self.conv3d = torch.nn.Conv3d(4, 32, (4,4,4), stride=(2,2,2), padding=(1,1,1))
        # self.layers = []
        self.down_in_dims = [in_channel, chans, 2*chans]
        self.down_out_dims = [chans, 2*chans, 4*chans]
        self.down_ksizes = [4, 4, 4]
        self.down_strides = [2, 2, 2]
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        print('down dims: ', self.down_out_dims)

        for i, (in_dim, out_dim, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            # print('3D CONV', end=' ')
             
            conv3d.append(nn.Sequential(
                SparseConvBlock(in_dim, out_dim, ksize, padding=padding),
                # nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_dim),
            ))

        self.conv3d = nn.ModuleList(conv3d)

        self.up_in_dims = [4*chans, 6*chans]
        self.up_bn_dims = [6*chans, 3*chans]
        self.up_out_dims = [4*chans, 2*chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        print('up dims: ', self.up_out_dims)

        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
             
            conv3d_transpose.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm3d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=3*chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        self.conv3d_transpose = nn.ModuleList(conv3d_transpose)
        self.up_bn = nn.ModuleList(up_bn)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv3d_transpose_layer, bn_layer) in enumerate(zip(self.conv3d_transpose, self.up_bn)):
            feat = conv3d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) #skip connection by concatenation
            feat = bn_layer(feat)

        feat = self.final_feature(feat)

        return feat


class ResNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(ResNet3D, self).__init__()
        # first layer: downsampling
        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        self.downsamp1_1 = nn.Sequential(
            SparseConvBlock(in_dim, out_dim, ksize, stride=stride, padding=padding),
        )
        self.downsamp1_2 = nn.Sequential(
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )
        
        # self.downsamp2_1 = nn.Sequential(
        #     SparseConvBlock(chans, chans, ksize, stride=stride, padding=padding),
        # )
        # self.downsamp2_2 = nn.Sequential(
        #     nn.BatchNorm3d(num_features=out_dim),
        #     nn.LeakyReLU(),
        # )

        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        self.res_block1_1 = self.generate_block_part1(in_dim, out_dim, ksize, stride, padding)
        self.res_block1_2 = self.generate_block_part2(in_dim, out_dim, ksize, stride, padding)
        self.res_block1_3 = self.generate_block_part3(in_dim, out_dim, ksize, stride, padding)

        self.res_block2_1 = self.generate_block_part1(in_dim, out_dim, ksize, stride, padding)
        self.res_block2_2 = self.generate_block_part2(in_dim, out_dim, ksize, stride, padding)
        self.res_block2_3 = self.generate_block_part3(in_dim, out_dim, ksize, stride, padding)

        self.res_block3_1 = self.generate_block_part1(in_dim, out_dim, ksize, stride, padding)
        self.res_block3_2 = self.generate_block_part2(in_dim, out_dim, ksize, stride, padding)
        self.res_block3_3 = self.generate_block_part3(in_dim, out_dim, ksize, stride, padding)

        # self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, padding)

        # self.res_block4_1 = self.generate_block_part1(in_dim, out_dim, ksize, stride, padding)
        # self.res_block4_2 = self.generate_block_part2(in_dim, out_dim, ksize, stride, padding)
        # self.res_block4_3 = self.generate_block_part3(in_dim, out_dim, ksize, stride, padding)

        # self.res_block5_1 = self.generate_block_part1(in_dim, out_dim, ksize, stride, padding)
        # self.res_block5_2 = self.generate_block_part2(in_dim, out_dim, ksize, stride, padding)
        # self.res_block5_3 = self.generate_block_part3(in_dim, out_dim, ksize, stride, padding)

        # self.res_block6_1 = self.generate_block_part1(in_dim, out_dim, ksize, stride, padding)
        # self.res_block6_2 = self.generate_block_part2(in_dim, out_dim, ksize, stride, padding)
        # self.res_block6_3 = self.generate_block_part3(in_dim, out_dim, ksize, stride, padding)

        self.lrelu = nn.LeakyReLU()

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        
    def generate_block(self, in_dim, out_dim, ksize, stride, padding):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            nn.BatchNorm3d(num_features=out_dim),
            )
        return block

    def generate_block_part1(self, in_dim, out_dim, ksize, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU())
    def generate_block_part2(self, in_dim, out_dim, ksize, stride, padding):
        return nn.Sequential(SparseConvBlock(in_dim, out_dim, ksize, stride=stride, padding=padding))
    def generate_block_part3(self, in_dim, out_dim, ksize, stride, padding):
        return nn.Sequential(
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm3d(num_features=out_dim))

    def forward(self, inputs, mask):
        feat = inputs
        feat, mask = self.downsamp1_1((feat, mask))
        feat = self.downsamp1_2(feat)

        # feat, mask = self.downsamp2_1((feat, mask))
        # feat = self.downsamp2_2(feat)

        # feat_before = feat
        # feat_after = self.res_block4_1(feat)
        # feat_after, mask = self.res_block4_2((feat_after, mask))
        # feat_after = self.res_block4_3(feat_after)
        # feat = feat_before + feat_after
        # feat = self.lrelu(feat)

        # feat_before = feat
        # feat_after = self.res_block5_1(feat)
        # feat_after, mask = self.res_block5_2((feat_after, mask))
        # feat_after = self.res_block5_3(feat_after)
        # feat = feat_before + feat_after
        # feat = self.lrelu(feat)

        # feat_before = feat
        # feat_after = self.res_block6_1(feat)
        # feat_after, mask = self.res_block6_2((feat_after, mask))
        # feat_after = self.res_block6_3(feat_after)
        # feat = feat_before + feat_after
        # feat = self.lrelu(feat)

        feat = self.final_feature(feat)
        
        return feat, mask

    
class Custom3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(Custom3D, self).__init__()

        self.pool3d = nn.MaxPool3d(2, stride=2, padding=0)
        
        # first layer: downsampling
        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1


        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        
        dims = [16, 32, 64]
        
        # dims = [8, 16, 32, 64]
        # for ind, dim in enumerate(dims):
        #     if ind==0:
        #         in_dim = 4
        #     else:
        #         in_dim = dims[ind-1]
        #     out_dim = dims[ind]
            
        in_dim = 4
        out_dim = dims[0]
        self.squeeze_block1 = self.generate_squeeze_block(in_dim, out_dim)
        self.res_block1 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        self.res_block2 = self.generate_residual_block(out_dim, out_dim, dilation=2)

        in_dim = dims[0]
        out_dim = dims[1]
        self.squeeze_block3 = self.generate_squeeze_block(in_dim, out_dim)
        self.res_block3 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        self.res_block4 = self.generate_residual_block(out_dim, out_dim, dilation=2)
        self.res_block5 = self.generate_residual_block(out_dim, out_dim, dilation=4)

        in_dim = dims[1]
        out_dim = dims[2]
        self.squeeze_block6 = self.generate_squeeze_block(in_dim, out_dim)
        self.res_block6 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        self.res_block7 = self.generate_residual_block(out_dim, out_dim, dilation=2)
        self.res_block8 = self.generate_residual_block(out_dim, out_dim, dilation=4)
        self.res_block9 = self.generate_residual_block(out_dim, out_dim, dilation=1)

        # upsample here
        self.res_block10 = self.generate_residual_block(out_dim, out_dim)
        self.res_block11 = self.generate_residual_block(out_dim, out_dim)
        
        # in_dim = dims[2]
        # out_dim = dims[3]
        # self.squeeze_block10 = self.generate_squeeze_block(in_dim, out_dim)
        # self.res_block10 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        # self.res_block11 = self.generate_residual_block(out_dim, out_dim, dilation=2)
        # self.res_block12 = self.generate_residual_block(out_dim, out_dim, dilation=4)
        # self.res_block13 = self.generate_residual_block(out_dim, out_dim, dilation=1)
        
        self.lrelu = nn.LeakyReLU()

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=dims[-1], out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        
    def generate_residual_block(self, in_dim, out_dim, dilation=1):
        if actually_standard_conv:
            block = nn.Sequential(
                Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
                LeakyRelu(),
                Pad(dilation),
                Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, dilation=dilation),
                BatchNorm(out_dim),
                LeakyRelu(),
                Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
            )
        else:
            block = nn.Sequential(
                Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
                LeakyRelu(),
                Pad(dilation),
                SparseConv(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, dilation=dilation),
                BatchNorm(out_dim),
                LeakyRelu(),
                Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
            )
        return block
    
    def generate_squeeze_block(self, in_dim, out_dim):
        block = nn.Sequential(
            Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            BatchNorm(out_dim),
        )
        return block

    def generate_block_part1(self, in_dim, out_dim, ksize, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU())
    def generate_block_part2(self, in_dim, out_dim, ksize, stride, padding):
        return nn.Sequential(SparseConvBlock(in_dim, out_dim, ksize, stride=stride, padding=padding))
    def generate_block_part3(self, in_dim, out_dim, ksize, stride, padding):
        return nn.Sequential(
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm3d(num_features=out_dim))

    def forward(self, feat, mask):
        
        def run_res_layer(feat, res, squeeze=None):
            if squeeze is not None:
                feat_before, _ = squeeze(feat)
            else:
                feat_before, _ = feat
            feat = res(feat)
            feat_after, new_mask = feat
            feat = feat_before + feat_after
            feat = self.lrelu(feat)
            return feat, new_mask

        feat, mask = self.pool3d(feat), self.pool3d(mask)
        feat, mask = run_res_layer((feat, mask), self.res_block1, self.squeeze_block1)
        feat, mask = run_res_layer((feat, mask), self.res_block2)
        
        feat, mask = self.pool3d(feat), self.pool3d(mask)
        feat, mask = run_res_layer((feat, mask), self.res_block3, self.squeeze_block3)
        feat, mask = run_res_layer((feat, mask), self.res_block4)
        feat, mask = run_res_layer((feat, mask), self.res_block5)
        
        feat, mask = self.pool3d(feat), self.pool3d(mask)
        feat, mask = run_res_layer((feat, mask), self.res_block6, self.squeeze_block6)
        feat, mask = run_res_layer((feat, mask), self.res_block7)
        feat, mask = run_res_layer((feat, mask), self.res_block8)
        feat, mask = run_res_layer((feat, mask), self.res_block9)

        # now up
        feat = F.interpolate(feat, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        feat, mask = run_res_layer((feat, mask), self.res_block10)
        
        feat = F.interpolate(feat, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        feat, mask = run_res_layer((feat, mask), self.res_block11)
        
        feat = self.final_feature(feat)

        if actually_standard_conv:
            print('making the mask all ones, and using conv instead of sparseconv')
            mask = torch.ones_like(feat[:,0:1])
        
        return feat, mask

class Sharp3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(Sharp3D, self).__init__()

        self.pool3d = nn.MaxPool3d(2, stride=2, padding=0)
        
        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        dims = [16, 32, 64]
            
        in_dim = 4
        out_dim = dims[0]
        self.squeeze_block1 = self.generate_squeeze_block(in_dim, out_dim)
        self.res_block1 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        self.res_block2 = self.generate_residual_block(out_dim, out_dim, dilation=2)

        in_dim = dims[0]
        out_dim = dims[1]
        self.squeeze_block3 = self.generate_squeeze_block(in_dim, out_dim)
        self.res_block3 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        self.res_block4 = self.generate_residual_block(out_dim, out_dim, dilation=2)
        self.res_block5 = self.generate_residual_block(out_dim, out_dim, dilation=4)
        # self.res_block6 = self.generate_residual_block(in_dim, out_dim, dilation=1)

        in_dim = dims[1]
        out_dim = dims[2]
        self.squeeze_block6 = self.generate_squeeze_block(in_dim, out_dim)
        self.res_block6 = self.generate_residual_block(in_dim, out_dim, dilation=1)
        self.res_block7 = self.generate_residual_block(out_dim, out_dim, dilation=2)
        self.res_block8 = self.generate_residual_block(out_dim, out_dim, dilation=4)
        self.res_block9 = self.generate_residual_block(out_dim, out_dim, dilation=1)

        self.lrelu = nn.LeakyReLU()

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=dims[-1], out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        
    def generate_residual_block(self, in_dim, out_dim, dilation=1):
        if actually_standard_conv:
            block = nn.Sequential(
                Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
                LeakyRelu(),
                Pad(dilation),
                Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, dilation=dilation),
                BatchNorm(out_dim),
                LeakyRelu(),
                Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
            )
        else:
            block = nn.Sequential(
                Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
                LeakyRelu(),
                Pad(dilation),
                SparseConv(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, dilation=dilation),
                BatchNorm(out_dim),
                LeakyRelu(),
                Conv(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1),
                BatchNorm(out_dim),
            )
        return block
    
    def generate_squeeze_block(self, in_dim, out_dim):
        block = nn.Sequential(
            Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            BatchNorm(out_dim),
        )
        return block

    def forward(self, feat, mask):
        
        def run_res_layer(feat, res, squeeze=None):
            if squeeze is not None:
                feat_before, _ = squeeze(feat)
            else:
                feat_before, _ = feat
            feat = res(feat)
            feat_after, new_mask = feat
            feat = feat_before + feat_after
            feat = self.lrelu(feat)
            return feat, new_mask

        feat, mask = self.pool3d(feat), self.pool3d(mask)
        feat, mask = run_res_layer((feat, mask), self.res_block1, self.squeeze_block1)
        feat, mask = run_res_layer((feat, mask), self.res_block2)
        
        feat, mask = run_res_layer((feat, mask), self.res_block3, self.squeeze_block3)
        feat, mask = run_res_layer((feat, mask), self.res_block4)
        feat, mask = run_res_layer((feat, mask), self.res_block5)
        # feat, mask = run_res_layer((feat, mask), self.res_block6)
        
        feat, mask = run_res_layer((feat, mask), self.res_block6, self.squeeze_block6)
        feat, mask = run_res_layer((feat, mask), self.res_block7)
        feat, mask = run_res_layer((feat, mask), self.res_block8)
        feat, mask = run_res_layer((feat, mask), self.res_block9)
        
        feat = self.final_feature(feat)

        if actually_standard_conv:
            print('making the mask all ones, and using conv instead of sparseconv')
            mask = torch.ones_like(feat[:,0:1])
        
        return feat, mask
    
    

if __name__ == "__main__":
    # net = Net3D(in_channel=4, pred_dim=32)
    net = ResNet3D(in_channel=4, pred_dim=32).cuda()
    print(net.named_parameters)
    inputs = torch.rand(2, 4, 128, 128, 32)
    time1 = time.time()
    out = net(inputs.cuda())
    print("time for dense:", time.time()-time1)
    print(out.size())


