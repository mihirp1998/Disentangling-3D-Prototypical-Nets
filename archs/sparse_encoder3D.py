import torch
import torch.nn as nn
import spconv
import time
 
def generate_sparse(feat, mask):
        B, C, D, H, W = list(feat.shape)
        coords = torch.nonzero(mask[:,0].int())
        # should be [N, 4]
        b, d, h, w = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        coords_flatten = w + h*W + d*H*W + b*D*H*W
        # should be [N]
        feat_flatten = torch.reshape(feat.permute(0,2,3,4,1), (-1, C)).float()
        feat_flatten = feat_flatten[coords_flatten]
        coords = coords.int()
        sparse_feat = spconv.SparseConvTensor(feat_flatten, coords.to('cpu'), [D, H, W], B)
        return sparse_feat

class SparseNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(SparseNet3D, self).__init__()
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
             
            conv3d += [
                spconv.SparseConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                ]

        # self.conv3d = nn.ModuleList(conv3d)
        self.conv3d = spconv.SparseSequential(*conv3d)

        self.up_in_dims = [4*chans, 6*chans]
        self.up_bn_dims = [6*chans, 3*chans]
        self.up_out_dims = [4*chans, 2*chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 #Note: this only holds for ksize=4 and stride=2!
        print('up dims: ', self.up_out_dims)

        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
             
            conv3d_transpose += [
                spconv.SparseConvTranspose3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_dim),
                ]
            up_bn.append(nn.BatchNorm1d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = spconv.SparseSequential(
                spconv.ToDense(),
                nn.Conv3d(in_channels=2*chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)
                )
        self.conv3d_transpose = spconv.SparseSequential(*conv3d_transpose)
        self.up_bn = nn.ModuleList(up_bn)


    def forward(self, feat, mask):
        mask = mask > 0.5
        feat = feat
        B, C, D, H, W = list(feat.shape)
        coords = torch.nonzero(mask[:,0].int())
        # should be [N, 4]
        print(list(coords.shape)[0])
        b, d, h, w = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        coords_flatten = w + h*W + d*H*W + b*D*H*W
        # should be [N]
        feat_flatten = torch.reshape(feat.permute(0,2,3,4,1), (-1, C)).float()
        feat_flatten = feat_flatten[coords_flatten]
        coords = coords.int()
        sparse_feat = spconv.SparseConvTensor(feat_flatten, coords, [D, H, W], B)
        sparse_feat = self.conv3d(sparse_feat)
        sparse_feat = self.conv3d_transpose(sparse_feat)
        sparse_feat = self.final_feature(sparse_feat)
        return sparse_feat
        

        '''
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
        '''
        # return feat



class SparseResNet3D(nn.Module):
    def __init__(self, in_channel, pred_dim, chans=64):
        super(SparseResNet3D, self).__init__()

        # self.conv3d = torch.nn.Conv3d(4, 32, (4,4,4), stride=(2,2,2), padding=(1,1,1))
        # self.layers = []
        block_num = 1
        do_subm = True

        # first lqyer - downsampling
        in_dim, out_dim, ksize, stride, padding = in_channel, chans, 4, 2, 1
        self.down_sampler = nn.Sequential(
            nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm3d(num_features=out_dim),
            nn.LeakyReLU(),
        )
        self.maxpooling = nn.MaxPool3d(2, stride=2)

        in_dim, out_dim, ksize, stride, padding = chans, chans, 3, 1, 1
        self.res_block1 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.lrelu_block1 = nn.LeakyReLU()
        self.res_block2 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.lrelu_block2 = nn.LeakyReLU()
        self.res_block3 = self.generate_block(in_dim, out_dim, ksize, stride, padding)
        self.lrelu_block3 = nn.LeakyReLU()
        # final 1x1x1 conv to get our desired pred_dim
        self.final_feature = nn.Conv3d(in_channels=chans, out_channels=pred_dim, kernel_size=1, stride=1, padding=0)


    def generate_block(self, in_dim, out_dim, ksize, stride, padding, do_subm=True):
        block = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, indice_key='subm0') if do_subm else spconv.SparseConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_dim),
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, indice_key="subm0"),
            nn.BatchNorm1d(num_features=out_dim),
            spconv.ToDense(),
            )
        return block


    def forward(self, feat, mask):
        # print("IS SPARSE")
        mask = self.maxpooling(mask)
        mask = mask > 0.5
        feat = self.down_sampler(feat)

        feat_before = feat
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block1(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block1(feat)

        feat_before = feat
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block2(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block2(feat)

        feat_before = feat
        feat_sparse = generate_sparse(feat, mask) # sparse feat
        feat_after = self.res_block3(feat_sparse) # dense feat
        feat = feat_before + feat_after
        feat = self.lrelu_block3(feat)

        feat = self.final_feature(feat)
        return feat


if __name__ == "__main__":
    shape = [2, 4, 128,128, 32] 
    sparse_featnet = SparseResNet3D(in_channel=4, pred_dim=32).cuda()
    print(sparse_featnet.named_parameters)
    inputs = torch.rand(shape) # N, C, D, H, W
    mask = torch.max(inputs, 1, keepdim=True)[0] 
    mask[:,:,0:64]*=0.0
    time1 = time.time()
    out = sparse_featnet(inputs.cuda(), mask.cuda())
    print("time for sparse:", time.time()-time1)
    print(out.size())


