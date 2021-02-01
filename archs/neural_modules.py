import torch
import torch.nn as nn
import time
# import hyperparams as hyp
import ipdb
st = ipdb.set_trace
# from utils_basic import *

class Net3D_NOBN(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(Net3D_NOBN, self).__init__()
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
                nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                # nn.BatchNorm3d(num_features=out_dim),
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
            # up_bn.append(nn.BatchNorm3d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=3*chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        self.conv3d_transpose = nn.ModuleList(conv3d_transpose)
        # self.up_bn = nn.ModuleList(up_bn)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon
        
        for i, conv3d_transpose_layer in enumerate(self.conv3d_transpose):
            feat = conv3d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) #skip connection by concatenation
            # feat = bn_layer(feat)

        feat = self.final_feature(feat)

        return feat


class ResNet3D_NOBN(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(ResNet3D_NOBN, self).__init__()
        # first lqyer - downsampling
        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 3,1,1
        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
            # nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )

        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        self.res_block1 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.res_block2 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, padding)

        self.lrelu = nn.LeakyReLU()

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        

    def generate_block(self, in_dim, out_dim, ksize, stride, padding):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # nn.BatchNorm3d(num_features=out_dim),
            )
        return block


    def forward(self, inputs):
        feat = inputs
        feat = self.down_sampler(feat)
        # st()
        feat_before = feat
        feat_after = self.res_block1(feat)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat_before = feat
        feat_after = self.res_block2(feat)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat_before = feat
        feat_after = self.res_block3(feat)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat = self.final_feature(feat)

        return feat


class ResNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=32):
        super(ResNet3D, self).__init__()
        # first lqyer - downsampling
        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 3,1,1
        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
            # nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )

        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        self.res_block1 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.res_block2 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, padding)

        self.lrelu = nn.LeakyReLU()

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
        

    def generate_block(self, in_dim, out_dim, ksize, stride, padding):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1),
            # nn.BatchNorm3d(num_features=out_dim),
            )
        return block


    def forward(self, inputs):
        feat = inputs
        feat = self.down_sampler(feat)

        feat_before = feat
        feat_after = self.res_block1(feat)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat_before = feat
        feat_after = self.res_block2(feat)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat_before = feat
        feat_after = self.res_block3(feat)
        feat = feat_before + feat_after
        feat = self.lrelu(feat)

        feat = self.final_feature(feat)
        return feat
if __name__ == "__main__":
    import os
    os.environ["MODE"] = "CLEVR_STA"
    net = ResNet3D(in_channel=32, pred_dim=32).cuda()
    print(net.named_parameters)
    inputs = torch.rand(1, 32, 16, 16, 16)
    time1 = time.time()
    out = net(inputs.cuda())
    st()
    print("time for dense:", time.time()-time1)
    print(out.size())


