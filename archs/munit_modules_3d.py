from torch import nn
import torch
import torch.nn.functional as F
import hyperparams as hyp
import archs.munit_modules as munit_modules
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

import ipdb
st = ipdb.set_trace


class AdaINGen3d(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen3d, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder3D(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder3D(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder3D(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = munit_modules.MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class Decoder3D(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder3D, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks3D(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv3DBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv3DBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        val = self.model(x)
        return val


class StyleEncoder3D(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder3D, self).__init__()
        self.model = []
        self.model += [Conv3DBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv3DBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv3DBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool3d(1)] # global average pooling
        self.model = nn.Sequential(*self.model)
        # self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.final_conv = nn.Conv2d(dim, style_dim, 1, 1, 0)
        self.output_dim = dim

    def forward(self, x):
        out = self.model(x)
        out = out.squeeze(-1) # so that 2d conv can be applied like in original implementation.
        out = self.final_conv(out)
        return out

class ContentEncoder3D(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder3D, self).__init__()
        self.model = []
        self.model += [Conv3DBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv3DBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks3D(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ResBlocks3D(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks3D, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock3D(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock3D(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock3D, self).__init__()

        model = []
        model += [Conv3DBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv3DBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out



class Conv3DBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv3DBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        self.padding = (padding, padding, padding, padding, padding, padding) # pad D,H,W
        self.doit = False
        # IMP: Reflect padding is giving not implemented. Changing it to constant padding for now.
        if pad_type == 'zero' or pad_type == 'reflect':
            pad_type = 'constant'
        
        self.pad_type = pad_type

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'ln':
            # Looks like adain is generic enough to work for 3d case too
            self.norm = munit_modules.LayerNorm(norm_dim)
        elif norm == 'adain':
            self.doit = True
            # Looks like adain is generic enough to work for 3d case too
            self.norm = munit_modules.AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.tmp_content = None
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            st() # Not being used probably. Remove after confirmation
            self.conv = munit_modules.SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        # st()
        padded_x = F.pad(x, self.padding, self.pad_type, 0)
        x = self.conv(padded_x)
        if self.norm:
            if self.doit:
                self.tmp_content = x
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



######################
## MUNIT Trainer
######################
class MUNITTrainer3D(nn.Module):
    def __init__(self):
        super(MUNITTrainer3D, self).__init__()
        # lr = hyperparameters['lr']
        # Initiate the networks
        hyperparameters = self.get_hyperparameters()
        self.hyperparameters = hyperparameters
        self.gen_a = AdaINGen3d(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen3d(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = munit_modules.MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = munit_modules.MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        # beta1 = hyperparameters['beta1']
        # beta2 = hyperparameters['beta2']
        # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        # gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        # self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(munit_modules.weights_init('kaiming'))
        self.dis_a.apply(munit_modules.weights_init('gaussian'))
        self.dis_b.apply(munit_modules.weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = torchvision.models.vgg16(pretrained=True)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def get_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['gan_w'] = hyp.munit_gan_w
        hyperparameters['recon_x_w'] = hyp.munit_recon_x_w
        hyperparameters['recon_s_w'] = hyp.munit_recon_s_w
        hyperparameters['recon_c_w'] = hyp.munit_recon_c_w
        hyperparameters['recon_x_cyc_w'] = hyp.munit_recon_x_cyc_w
        hyperparameters['vgg_w'] = hyp.munit_vgg_w
        hyperparameters['input_dim_b'] = hyp.munit_input_dim_b
        hyperparameters['input_dim_a'] = hyp.munit_input_dim_a
        hyperparameters['display_size'] = hyp.munit_display_size
        hyperparameters['gen'] = hyp.munit_gen
        hyperparameters['dis'] = hyp.munit_dis
        return hyperparameters


    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = self.s_a
        s_b = self.s_b
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba
    
    def gen_update_paired(self, x_a, x_b, hyperparameters):

        '''
        For paired data, we will just use gen_a as there is no such thing as domain right now.
        '''
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_a.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime) # should match x_a
        x_b_recon = self.gen_a.decode(c_b, s_b_prime) # should match x_b
        # st()
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a_prime) # should match x_a
        x_ab = self.gen_a.decode(c_a, s_b_prime) # should match x_b
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_a.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_a.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        self.loss_gen_recon_x_b_to_a = self.recon_criterion(x_ba, x_a)
        self.loss_gen_recon_x_a_to_b = self.recon_criterion(x_ab, x_b)

        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # TODO: apply viewpred here?
        
        # total loss
        
        self.loss_gen_total = hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b_to_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a_to_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b
                              
        # st()
        return self.loss_gen_total, x_ba, x_ab, x_a_recon, x_b_recon, x_aba, x_bab, (s_a_prime,  s_b_prime), (c_a,  c_b)

    def gen_update(self, x_a, x_b, hyperparameters):
        # self.gen_opt.zero_grad()
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        # encodess
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # TODO: apply viewpred here?
        st() # Remove this st() after addressing this TODO.
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        # self.loss_gen_total.backward()
        # self.gen_opt.step()
        return self.loss_gen_total

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = munit_modules.vgg_preprocess(img)
        target_vgg = munit_modules.vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        st() # This has not been fixed for 3d case yet.
        self.eval()
        s_a1 = self.s_a
        s_b1 = self.s_b
        s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        # TODO: apply viewpred here?
        st() # Remove this st() after addressing this TODO.
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        
        # self.loss_dis_total.backward()
        # self.dis_opt.step()
        return self.loss_dis_total

    # def update_learning_rate(self):
    #     if self.dis_scheduler is not None:
    #         self.dis_scheduler.step()
    #     if self.gen_scheduler is not None:
    #         self.gen_scheduler.step()

    # def resume(self, checkpoint_dir, hyperparameters):
    #     # Load generators
    #     last_model_name = get_model_list(checkpoint_dir, "gen")
    #     state_dict = torch.load(last_model_name)
    #     self.gen_a.load_state_dict(state_dict['a'])
    #     self.gen_b.load_state_dict(state_dict['b'])
    #     iterations = int(last_model_name[-11:-3])
    #     # Load discriminators
    #     last_model_name = get_model_list(checkpoint_dir, "dis")
    #     state_dict = torch.load(last_model_name)
    #     self.dis_a.load_state_dict(state_dict['a'])
    #     self.dis_b.load_state_dict(state_dict['b'])
    #     # Load optimizers
    #     state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
    #     self.dis_opt.load_state_dict(state_dict['dis'])
    #     self.gen_opt.load_state_dict(state_dict['gen'])
    #     # Reinitilize schedulers
    #     self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
    #     self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
    #     print('Resume from iteration %d' % iterations)
    #     return iterations

    # def save(self, snapshot_dir, iterations):
    #     # Save generators, discriminators, and optimizers
    #     gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
    #     dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
    #     opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
    #     torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
    #     torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
    #     torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

class MUNITTrainer3D_Simple(nn.Module):
    def __init__(self):
        super(MUNITTrainer3D_Simple, self).__init__()
        # lr = hyperparameters['lr']
        # Initiate the networks
        hyperparameters = self.get_hyperparameters()
        self.hyperparameters = hyperparameters
        self.gen_a = AdaINGen3d_Simple(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
    
    def get_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['gan_w'] = hyp.munit_gan_w
        hyperparameters['recon_x_w'] = hyp.munit_recon_x_w
        hyperparameters['recon_s_w'] = hyp.munit_recon_s_w
        hyperparameters['recon_c_w'] = hyp.munit_recon_c_w
        hyperparameters['recon_x_cyc_w'] = hyp.munit_recon_x_cyc_w
        hyperparameters['vgg_w'] = hyp.munit_vgg_w
        hyperparameters['input_dim_b'] = hyp.munit_input_dim_b
        hyperparameters['input_dim_a'] = hyp.munit_input_dim_a
        hyperparameters['display_size'] = hyp.munit_display_size
        hyperparameters['gen'] = hyp.munit_gen
        hyperparameters['dis'] = hyp.munit_dis
        return hyperparameters


    def recon_criterion(self, input, target):
        return torch.mean((input - target)**2)

    
    def gen_update_paired(self, x_a, x_b, hyperparameters):

        '''
        For paired data, we will just use gen_a as there is no such thing as domain right now.
        '''
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_a.encode(x_b)
        # decode (within domain)
        x_a_recon, adin_cs_a = self.gen_a.decode(c_a, s_a_prime) # should match x_a
        x_b_recon, adin_cs_b = self.gen_a.decode(c_b, s_b_prime) # should match x_b
        # st()
        # decode (cross domain)
        x_ba, _ = self.gen_a.decode(c_b, s_a_prime) # should match x_a
        x_ab, _ = self.gen_a.decode(c_a, s_b_prime) # should match x_b
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_a.encode(x_ab)
        # decode again (if needed)
        x_aba, _ = self.gen_a.decode(c_a_recon, s_a_prime)
        x_bab, _ = self.gen_a.decode(c_b_recon, s_b_prime) 

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)
        # TODO: apply viewpred here?
        
        # total loss
        
        self.loss_gen_total = hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b
                              

        return self.loss_gen_total, x_ba, x_ab, x_a_recon, x_b_recon, x_aba, x_bab, (s_a_prime,  s_b_prime), (c_a,  c_b),(adin_cs_a,adin_cs_b)

class AdaINGen3d_Simple(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen3d_Simple, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        # st()
        self.enc_style = StyleEncoder3D_Simple(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
        # content encoder
        self.enc_content = ContentEncoder3D_Simple(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder3D_Simple(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = munit_modules.MLP_Simple(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
        # st()

    def encode(self, tensor):
        # encode an image to its content and style codes
        # st()
        style_fake = self.enc_style(tensor)
        content = self.enc_content(tensor)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        # st()
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        tensor = self.dec(content)

        content_adin = self.dec.model[0].tmp_content
        style_adin = adain_params
        return tensor,(content_adin,style_adin)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params



class StyleEncoder3D_Simple(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder3D_Simple, self).__init__()
        self.model = []
        self.model += [Conv3DBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        # for i in range(1):
        self.model += [Conv3DBlock(dim, style_dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            # dim *= 2        
        # for i in range(2):
        #     self.model += [Conv3DBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        #     dim *= 2
        # for i in range(n_downsample - 2):
        #     self.model += [Conv3DBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool3d(1)] 
        self.model = nn.Sequential(*self.model)
        # st()
        # self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        # self.final_conv = nn.Conv2d(dim, style_dim, 1, 1, 0)
        self.output_dim = dim

    def forward(self, x):
        out = self.model(x)
        return out


class ContentEncoder3D_Simple(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder3D_Simple, self).__init__()
        self.model = []
        self.model += [Conv3DBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        # for i in range(1):
        self.model += [Conv3DBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2
        # residual blocks
        # st()
        # self.model += [ResBlocks3D(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim
    def forward(self, x):
        val =  self.model(x)
        return val




class Decoder3D_Simple(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder3D_Simple, self).__init__()

        self.model = []
        # st()
        # AdaIN residual blocks
        # self.model += [ResBlocks3D(n_res, dim, res_norm, activ, pad_type=pad_type)]
        self.model += [Conv3DBlock(dim, dim, 3, 1, 1, norm=res_norm, activation=activ, pad_type=pad_type)]
        # self.model += [Conv3DBlock(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        
        self.model += [nn.Upsample(scale_factor=2)]
        # dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv3DBlock(dim, output_dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        val = self.model(x)
        return val