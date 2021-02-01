# import tensorflow as tf
import ipdb 
st = ipdb.set_trace
from lib_classes import Nel_Utils as nlu
import torch
import torchvision.transforms
import cv2
import os
import numpy as np
from matplotlib import cm
import hyperparams as hyp
import utils_geom
import matplotlib
import imageio
from itertools import combinations
from tensorboardX import SummaryWriter


from utils_basic import *
import utils_basic
from sklearn.decomposition import PCA
import ipdb 
st = ipdb.set_trace
EPS = 1e-6
MAXWIDTH = 1800

'''color conversion in torch'''
from skimage.color import (rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
						   rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def _convert(input_, type_):
	return {
		'float': input_.float(),
		'double': input_.double(),
	}.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
	def apply_transform(input_):
		to_squeeze = (input_.dim() == 3)
		device = input_.device
		input_ = input_.cpu()
		input_ = _convert(input_, in_type)

		if to_squeeze:
			input_ = input_.unsqueeze(0)

		input_ = input_.permute(0, 2, 3, 1).numpy()
		transformed = transform(input_)
		output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
		if to_squeeze:
			output = output.squeeze(0)
		output = _convert(output, out_type)
		return output.to(device)
	return apply_transform


def _generic_transform_sk_3d(transform, in_type='', out_type=''):
	def apply_transform_individual(input_):
		device = input_.device
		input_ = input_.cpu()
		input_ = _convert(input_, in_type)

		input_ = input_.permute(1, 2, 0).detach().numpy()
		transformed = transform(input_)
		output = torch.from_numpy(transformed).float().permute(2, 0, 1)
		output = _convert(output, out_type)
		return output.to(device)

	def apply_transform(input_):
		to_stack = []
		for image in input_:
			to_stack.append(apply_transform_individual(image))
		return torch.stack(to_stack)
	return apply_transform


# --- Cie*LAB ---
rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type='double', out_type='float')
# --- YUV ---
rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
# --- YCbCr ---
rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
# --- HSV ---
rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)
# --- XYZ ---
rgb_to_xyz = _generic_transform_sk_4d(rgb2xyz)
xyz_to_rgb = _generic_transform_sk_3d(xyz2rgb, in_type='double', out_type='float')
# --- HED ---
rgb_to_hed = _generic_transform_sk_4d(rgb2hed)
hed_to_rgb = _generic_transform_sk_3d(hed2rgb, in_type='double', out_type='float')

'''end color conversion in torch'''

def preprocess_color_tf(x):
	import tensorflow as tf
	return tf.cast(x,tf.float32) * 1./255 - 0.5

def preprocess_color(x):
	if type(x).__module__ == np.__name__:
		return x.astype(np.float32) * 1./255 - 0.5
	else:
		return x.float() * 1./255 - 0.5

def preprocess_depth(x):
	if type(x).__module__ == np.__name__:
		return x.astype(np.float32) * 1./100 - 0.5
	else:
		return x.float() * 1./100 - 0.5	

def pca_embed(emb, keep):
	## emb -- [S,H/2,W/2,C]
	## keep is the number of principal components to keep
	## Helper function for reduce_emb.
	emb = emb + EPS
	#emb is B x C x H x W
	emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

	emb_reduced = list()

	B, H, W, C = np.shape(emb)
	for img in emb:
		if np.isnan(img).any():
			emb_reduced.append(np.zeros([H, W, keep]))
			continue

		pixelskd = np.reshape(img, (H*W, C))
		P = PCA(keep)
		P.fit(pixelskd)
		pixels3d = P.transform(pixelskd)
		out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
		if np.isnan(out_img).any():
			emb_reduced.append(np.zeros([H, W, keep]))
			continue

		emb_reduced.append(out_img)

	emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)

	return torch.from_numpy(emb_reduced).permute(0, 3, 1, 2)

def pca_embed_together(emb, keep):
	## emb -- [S,H/2,W/2,C]
	## keep is the number of principal components to keep
	## Helper function for reduce_emb.
	emb = emb + EPS
	#emb is B x C x H x W
	emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

	B, H, W, C = np.shape(emb)
	if np.isnan(emb).any():
		out_img = torch.zeros(B, keep, H, W)

	pixelskd = np.reshape(emb, (B*H*W, C))
	P = PCA(keep)
	P.fit(pixelskd)
	pixels3d = P.transform(pixelskd)
	out_img = np.reshape(pixels3d, [B,H,W,keep]).astype(np.float32)
	if np.isnan(out_img).any():
		out_img = torch.zeros(B, keep, H, W)
	return torch.from_numpy(out_img).permute(0, 3, 1, 2)

def reduce_emb(emb, inbound=None, together=False):
	## emb -- [S,C,H/2,W/2], inbound -- [S,1,H/2,W/2]
	## Reduce number of chans to 3 with PCA. For vis.
	# S,H,W,C = emb.shape.as_list()
	S, C, H, W = list(emb.size())
	keep = 3

	if together:
		reduced_emb = pca_embed_together(emb, keep)
	else:
		reduced_emb = pca_embed(emb, keep) #not im

	reduced_emb = normalize(reduced_emb) - 0.5
	if inbound is not None:
		emb_inbound = emb*inbound
	else:
		emb_inbound = None

	return reduced_emb, emb_inbound

def get_feat_pca(feat):
	B, C, D, W = list(feat.size())
	# feat is B x C x D x W. If 3D input, average it through Height dimension before passing into this function.

	pca, _ = reduce_emb(feat, inbound=None, together=True)
	# pca is B x 3 x W x D
	return pca

def convert_occ_to_height(occ, reduce_axis=3):
	B, C, D, H, W = list(occ.shape)
	assert(C==1)
	# note that height increases DOWNWARD in the tensor
	# (like pixel/camera coordinates)
	
	G = list(occ.shape)[reduce_axis]
	values = torch.linspace(float(G), 1.0, steps=G).type(torch.FloatTensor).cuda()
	if reduce_axis==2:
		# frontal view
		values = values.view(1, 1, G, 1, 1)
	elif reduce_axis==3:
		# top view
		values = values.view(1, 1, 1, G, 1)
	elif reduce_axis==4:
		# lateral view
		values = values.view(1, 1, 1, 1, G)
	else:
		assert(False) # you have to reduce one of the spatial dims (2-4)
	values = torch.max(occ*values, dim=reduce_axis)[0]/float(G)
	# values = values.view([B, C, D, W])
	return values

def gif_and_tile(ims, just_gif=False):
	S = len(ims) 
	# ims is S X B X H X W X C
	# i want a gif in the left, and the tiled frames on the right
	# for the gif tool, this means making a B x S x H x W tensor
	# where the leftmost part is sequential and the rest is tiled
	gif = torch.stack(ims, dim=1)
	if just_gif:
		return gif
	til = torch.cat(ims, dim=2)
	til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
	im = torch.cat([gif, til], dim=3)
	return im

def back2color(i, blacken_zeros=False):
	if blacken_zeros:
		const = torch.tensor([-0.5])
		i = torch.where(i==0.0, const.cuda() if i.is_cuda else const, i)
		return back2color(i)
	else:
		return ((i+0.5)*255).type(torch.ByteTensor)

def colorize(d):
	# this does not work properly yet
	
	# # d is C x H x W or H x W
	# if d.ndim==3:
	#     d = d.squeeze(dim=0)
	# else:
	#     assert(d.ndim==2)

	if d.ndim==2:
		d = d.unsqueeze(dim=0)
	else:
		assert(d.ndim==3)
	# copy to the three chans
	d = d.repeat(3, 1, 1)
	return d
	
	# d = d.cpu().detach().numpy()
	# # move channels out to last dim
	# # d = np.transpose(d, [0, 2, 3, 1])
	# # d = np.transpose(d, [1, 2, 0])
	# print(d.shape)
	# d = cm.inferno(d)[:, :, 1:] # delete the alpha channel
	# # move channels into dim0
	# d = np.transpose(d, [2, 0, 1])
	# print_stats(d, 'colorize_out')
	# d = torch.from_numpy(d)
	# return d

def oned2inferno(d, norm=True):
	# convert a 1chan input to a 3chan image output

	# if it's just B x H x W, add a C dim
	if d.ndim==3:
		d = d.unsqueeze(dim=1)
	# d should be B x C x H x W, where C=1
	B, C, H, W = list(d.shape)
	assert(C==1)

	if norm:
		d = normalize(d)
		
	rgb = torch.zeros(B, 3, H, W)
	for b in range(B):
		rgb[b] = colorize(d[b])

	rgb = (255.0*rgb).type(torch.ByteTensor)

	# rgb = tf.cast(255.0*rgb, tf.uint8)
	# rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
	# rgb = tf.expand_dims(rgb, axis=0)
	return rgb

def xy2mask(xy, H, W, norm=False):
	# xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
	# proto is B x H x W x 1, showing how big to make the mask
	# returns a mask the same size as proto, with a 1 at each specified xy
	B = list(xy.shape)[0]
	if norm:
		# convert to pixel coords
		x, y = torch.unbind(xy, axis=2)
		x = x*float(W)
		y = y*float(H)
		xy = torch.stack(xy, axis=2)
		
	mask = torch.zeros([B, 1, H, W], dtype=torch.float32, device=torch.device('cuda'))
	for b in range(B):
		mask[b] = xy2mask_single(xy, H, W)
	return mask

def xy2mask_single(xy, H, W):
	# xy is N x 2
	x, y = torch.unbind(xy, axis=1)
	x = x.long()
	y = y.long()

	x = torch.clamp(x, 0, W-1)
	y = torch.clamp(y, 0, H-1)
	
	inds = sub2ind(H, W, y, x)

	valid = (inds > 0).byte() & (inds < H*W).byte()
	inds = inds[torch.where(valid)]

	mask = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
	mask[inds] = 1.0
	mask = torch.reshape(mask, [1,H,W])
	return mask

def get_unps_vis(unps, occs,dim=4):
	B, S, C, D, H, W = list(unps.shape)
	occs = occs.repeat(1, 1, C, 1, 1, 1)
	unps = reduce_masked_mean(unps, occs, dim=dim)
	# unps is B x S x W x D x C
	return unps

def get_unps_3d(unps, occs):
	B, S, C, D, H, W = list(unps.shape)
	occs = occs.repeat(1, 1, C, 1, 1, 1)
	unps = unps*occs
	# unps is B x S x W x D x C
	return unps



def draw_rect_on_image(rgb_torch, box, scale,negative= False):
	C, H, W = list(rgb_torch.shape)
	assert(C==3)
	rgb_torch = back2color(rgb_torch)

	box = np.array([int(i) for i in box])

	rgb = rgb_torch.cpu().numpy()

	rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
	rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

	start_point = box*scale
	end_point = start_point + hyp.max.searchRegion*scale 
	
	if negative:
		# red
		color = (0, 255, 0) 	
	else:
		# blue
		color = (255, 0, 0) 

	thickness = 0

	rgb = rgb.astype(np.uint8)
	rgb = cv2.rectangle(rgb, tuple(start_point), tuple(end_point), color, thickness) 


	out = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
	out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
	out = torch.unsqueeze(out, dim=0)
	out = preprocess_color(out)
	out = torch.reshape(out, [1, C, H, W])
	return out






class Summ_writer(object):
	def __init__(self, writer, global_step, set_name, fps=8,log_freq=None):
		self.global_step = global_step
		self.writer = writer
		self.fps = fps
		self.maxwidth = MAXWIDTH
		if log_freq is None:
			if set_name == "train":
				self.log_freq = hyp.log_freq_train
			elif set_name == "val":
				self.log_freq = hyp.log_freq_val
			elif set_name == "test":
				self.log_freq = hyp.log_freq_test
			else:
				assert False
		else:
			self.log_freq = log_freq
		# if hyp.hard_vis:
		# 	self.eval_mine = nlu.Eval_Mining()
		# self.mbr32 = cross_corr.meshgrid_based_rotation(32,32,32, angleIncrement=5)
		# self.mbr16 = cross_corr.meshgrid_based_rotation(16,16,16, angleIncrement=5)
		if hyp.set_fake:
			self.save_this = False
		else:
			self.save_this = (self.global_step % self.log_freq == 0)
		
	
	def summ_text(self, name, text):
		assert isinstance(text,str)
		self.writer.add_text(name, text , global_step=self.global_step)



	def summ_gif(self, name, tensor, blacken_zeros=False):
		# tensor should be in B x S x C x H x W
		
		assert tensor.dtype in {torch.uint8,torch.float32}
		shape = list(tensor.shape)
		# assert len(shape) in {4,5}
		# assert shape[4] in {1,3}

		# if len(shape) == 4:
		#     tensor = tensor.unsqueeze(dim=0)

		if tensor.dtype == torch.float32:
			tensor = back2color(tensor, blacken_zeros=blacken_zeros)

		#tensor = tensor.data.numpy()
		#tensor = np.transpose(tensor, axes=[0, 1, 4, 2, 3])

		# tensor = tensor.permute(0, 1, 4, 2, 3) #move the color channel to dim=2

		# tensor = tensor.transpose(2, 4).transpose(3, 4)

		video_to_write = tensor[0:1] #only keep the first if batch > 1 

		self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)

	def summ_points_on_mem(self, name, rgbR, points):
		C, H, W = list(rgbR.shape)
		# st()
		img = back2color(rgbR)
		for i in range(points.shape[0]):
			img[:, points[i,2], points[i,0]] = torch.tensor([255,0,0])

		# rgbR[:,points[:, 0], points[:, 2]] = torch.zeros(3)
		# st()
		self.summ_rgb(name, img.unsqueeze(0))


	def draw_cornerEnds_on_unp(self,rgb_torch, boxes,scores,tids):
		C, H, W = list(rgb_torch.shape)
		assert(C==3)
		rgb_torch = back2color(rgb_torch)
		rgb = rgb_torch.cpu().numpy()
		rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
		rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		color = (0, 255, 0) 	
		thickness = 0
		rgb = rgb.astype(np.uint8)

		for i,box in enumerate(boxes):
			if scores[i] > 0:
				lower, upper = torch.unbind(box)
				xmin,ymin,zmin = [int(i) for i in lower.detach().cpu().numpy()]
				xmax,ymax,zmax = [int(i) for i in upper.detach().cpu().numpy()]
				rgb = cv2.rectangle(rgb, (xmin,zmin), (xmax,zmax), color, thickness) 
		out = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
		out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
		out = torch.unsqueeze(out, dim=0)
		out = preprocess_color(out)
		out = torch.reshape(out, [1, C, H, W])
		return out



	def summ_hardmines(self,name,val,mbr_unpr):
		if self.save_this:
			negative = False
			if name == "negative":
				negative = True			
			posPair,[topkImg,topkD,topkH,topkW,topkR],ranks,[unpRs_e,unpRs_g],[ob_visual_2d_e,ob_visual_2d_g],current_index = val
			unpR_e_boxed,unpR_g_boxed = nlu.visualize_hard_mining(posPair,[topkImg,topkD,topkH,topkW,topkR],ranks,[unpRs_e,unpRs_g],[ob_visual_2d_e,ob_visual_2d_g],current_index,mbr_unpr,negative=negative)
			ns = "hardmining/"
			self.summ_rgbs(ns + name, [unpR_g_boxed, unpR_e_boxed])


	def summ_evalmines(self,name,val,mbr_unpr):
		if self.save_this:
			top_g,selected_e,unp_ge,vis2D_ge= val
			unps_boxed_e_gs = nlu.visualize_eval_mining(top_g,selected_e,unp_ge,vis2D_ge,self,mbr_unpr)
			ns = "hardmining/"
			# unps_boxed_e_gs = torch.cat(unps_boxed_e_gs,dim=2)
			# st()
			unps_boxed_e_gs = torch.cat(unps_boxed_e_gs,dim=3)
			self.summ_rgb(ns + name,unps_boxed_e_gs)



	def summ_rgbs(self, name, ims, blacken_zeros=False):
		if self.save_this:

			ims = gif_and_tile(ims)
			vis = ims

			B, S, C, H, W = list(vis.shape)

			if int(W) > self.maxwidth:
				vis = vis[:,:,:,:self.maxwidth]

			self.summ_gif(name, vis, blacken_zeros)

	def summ_rgb(self, name, ims, blacken_zeros=False):
		if self.save_this:
			assert ims.dtype in {torch.uint8,torch.float32}

			if ims.dtype == torch.float32:
				ims = back2color(ims, blacken_zeros)

			#ims is B x C x H x W
			vis = ims[0:1] # just the first one
			B, C, H, W = list(vis.shape)

			if int(W) > self.maxwidth:
				vis = vis[:,:,:,:self.maxwidth]

			self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

			# self.writer.add_images(name, vis, global_step=self.global_step, dataformats='NCHW')
	
	def summ_occs(self, name, occs, reduce_axes=[3]):
		if self.save_this:
			B, C, D, H, W = list(occs[0].shape)
			for reduce_axis in reduce_axes:
				heights = [convert_occ_to_height(occ, reduce_axis=reduce_axis) for occ in occs]
				self.summ_oneds(name=('%s_ax%d' % (name, reduce_axis)), ims=heights, norm=False)
			
	def summ_occ(self, name, occ, reduce_axes=[3]):
		if self.save_this:
			B, C, D, H, W = list(occ.shape)
			for reduce_axis in reduce_axes:
				height = convert_occ_to_height(occ, reduce_axis=reduce_axis)
				self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False)
	def summ_diff_tensor(self, name, feat_diff, reduce_axes=[3]):
		if self.save_this:
			feat_diff = torch.abs(feat_diff)
			feat_diff = torch.sum(feat_diff,dim=1).unsqueeze(1)
			feat_diff = utils_basic.l2_normalize(feat_diff)
			# st()
			B, C, D, H, W = list(feat_diff.shape)
			for reduce_axis in reduce_axes:
				height = convert_occ_to_height(feat_diff, reduce_axis=reduce_axis)
				self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False)


	def summ_oneds(self, name, ims, is3D=False, norm=True,maxdepth=None):
		if self.save_this:
			# if is3D:
			#     ims = [im.transpose(1, 2) for im in ims]

			if len(ims) != 1: #sequence
				im = gif_and_tile(ims)
			else:
				im = torch.stack(ims, dim=1) #single frame

			B, S, C, H, W = list(im.shape)
			if maxdepth is not None:
				im = im.clamp_(0,maxdepth)
			if norm:
				# normalize before oned2inferno,
				# so that the ranges are similar within B across S
				im = normalize(im)

			im = im.view(B*S, C, H, W)
			vis = oned2inferno(im, norm=norm)
			vis = vis.view(B, S, 3, H, W)
			if W > self.maxwidth:
				vis = vis[...,:self.maxwidth]

			# writer.add_images(name + "im", vis[:,0], global_step=global_step, dataformats='NCHW')
			self.summ_gif(name, vis)

	def summ_oned(self, name, im, is3D=False, norm=True):
		if self.save_this:
			# if is3D:
			#     im = im.transpose(2, 3)

			B, C, H, W = list(im.shape)
			im = im[0:1] # just the first one
			assert(C==1)
			vis = oned2inferno(im, norm=norm)
			# vis = vis.view(B, 3, H, W)
			if W > self.maxwidth:
				vis = vis[...,:self.maxwidth]
			# self.writer.add_images(name, vis, global_step=self.global_step, dataformats='NCHW')
			self.summ_rgb(name, vis, blacken_zeros=False)
			# writer.add_images(name + "_R", vis[:,0:1], global_step=global_step, dataformats='NCHW')
			# writer.add_images(name + "_G", vis[:,1:2], global_step=global_step, dataformats='NCHW')
			# writer.add_images(name + "_B", vis[:,2:3], global_step=global_step, dataformats='NCHW')
		
	def summ_unps(self, name, unps, occs):
		if self.save_this:
			unps = torch.stack(unps, dim=1)
			occs = torch.stack(occs, dim=1)
			B, S, C, D, H, W = list(unps.shape)
			occs = occs.repeat(1, 1, C, 1, 1, 1)
			unps = reduce_masked_mean(unps, occs, dim=4)
			unps = torch.unbind(unps, dim=1) #should be S x B x W x D x C
			# unps = [unp.transpose(1, 2) for unp in unps] #rotate 90 degree counter-clockwise
			self.summ_rgbs(name=name, ims=unps, blacken_zeros=True) 

	def summ_unp(self, name, unp, occ):
		if self.save_this:
			B, C, D, H, W = list(unp.shape)
			occ = occ.repeat(1, C, 1, 1, 1)
			unp = reduce_masked_mean(unp, occ, dim=3)
			# unp = [unp.transpose(1, 2) for unp in unp] #rotate 90 degree counter-clockwise
			self.summ_rgb(name=name, ims=unp, blacken_zeros=True) 

	def summ_feats(self, name, feats, valids=None, pca=True):
		if self.save_this:
			feats = torch.stack(feats, dim=1)
			# feats leads with B x S x C

			if feats.ndim==6:
				# feats is B x S x C x D x H x W
				if valids is None:
					feats = torch.mean(feats, dim=4)
				else:
					valids = torch.stack(valids, dim=1)
					valids = valids.repeat(1, 1, feats.size()[2], 1, 1, 1)
					feats = reduce_masked_mean(feats, valids, dim=4)

			B, S, C, D, W = list(feats.size())

			if not pca:
				# feats leads with B x S x C
				feats = torch.mean(torch.abs(feats), dim=2, keepdims=True)
				# feats leads with B x S x 1
				
				# feats is B x S x D x W
				feats = torch.unbind(feats, dim=1)
				# feats is a len=S list, each element of shape B x W x D
				# # make "forward" point up, and make "right" point right
				# feats = [feat.transpose(1, 2) for feat in feats]
				self.summ_oneds(name=name, ims=feats, norm=True)

			else: #pca

				__p = lambda x: pack_seqdim(x, B)
				__u = lambda x: unpack_seqdim(x, B)

				feats_ = __p(feats)
				feats_pca_ = get_feat_pca(feats_)

				feats_pca = __u(feats_pca_)

				self.summ_rgbs(name=name, ims=torch.unbind(feats_pca, dim=1))

	def summ_feat(self, name, feat, valid=None, pca=True):
		if self.save_this:
			if feat.ndim==5: #B x C x D x H x W
				if valid is None:
					feat = torch.mean(feat, dim=3)
				else:
					feat = reduce_masked_mean(feat, valid, dim=3)
					
			B, C, D, W = list(feat.shape)

			if not pca:
				feat = torch.mean(torch.abs(feat), dim=1, keepdims=True)
				# feat is B x 1 x D x W
				self.summ_oned(name=name, im=feat, norm=True)

			else:
				feat_pca = get_feat_pca(feat)
				self.summ_rgb(name, feat_pca)

	def summ_scalar(self, name, value):
		# if self.save_this:
		self.writer.add_scalar(name, value, global_step=self.global_step)

	def summ_scalars(self, name, value):
		# value is a dict
		# if self.save_this:
		self.writer.add_scalars(name, value, global_step=self.global_step)


	def summ_diff(self, name, value):
		# value is a dict
		# if self.save_this:
		self.writer.add_scalars(name, value, global_step=self.global_step)

	def summ_box(self, name, rgbR, boxes_camR, scores, tids, pix_T_cam, only_return=False):
		B, C, H, W = list(rgbR.shape)
		corners_camR = utils_geom.transform_boxes_to_corners(boxes_camR)
		return self.summ_box_by_corners(name, rgbR, corners_camR, scores, tids, pix_T_cam, only_return=only_return)

	def summ_box_mem_on_mem(self, name, rgbR, boxes_theta, scores, tids, only_return=False,text=False):
		# rgb is B x H x W x C
		# corners is B x N x 8 x 3
		# scores is B x N
		# tids is B x N
		# pix_T_cam is B x 4 x 4
		# st()
		corners = utils_geom.transform_boxes_to_corners(boxes_theta)
		corners_end = nlu.get_ends_of_corner(corners)
		B, C, H, W = list(rgbR.shape)
		# st()

		boxes_vis = self.draw_cornerEnds_on_unp(rgbR[0],
											   corners_end[0],
											   scores[0],
											   tids[0])
		if not only_return:
			self.summ_rgb(name, boxes_vis)
		return boxes_vis		
		# B, C, H, W = list(rgbR.shape)
		# boxes_vis = self.draw_boxes3D_mem_on_mem(rgbR,
		# 									   corners,
		# 									   scores,
		# 									   tids,text=text)
		# if not only_return:
		# 	self.summ_rgb(name, boxes_vis)
		# return boxes_vis

	def summ_box_mem_on_unp(self, name, rgbR, corners_end, scores, tids,only_return=False):
		# rgb is B x H x W x C
		# corners is B x N x 8 x 3
		# scores is B x N
		# tids is B x N
		# pix_T_cam is B x 4 x 4
		B, C, H, W = list(rgbR.shape)

		boxes_vis = self.draw_cornerEnds_on_unp(rgbR[0],
											   corners_end[0],
											   scores[0],
											   tids[0])
		if not only_return:
			self.summ_rgb(name, boxes_vis)
		return boxes_vis




	def summ_box_by_corners(self, name, rgbR, corners, scores, tids, pix_T_cam, only_return=False):
		# rgb is B x H x W x C
		# corners is B x N x 8 x 3 
		# scores is B x N
		# tids is B x N
		# pix_T_cam is B x 4 x 4
		# st()
		B, C, H, W = list(rgbR.shape)
		boxes_vis = self.draw_corners_on_image(rgbR,
											   corners,
											   scores,
											   tids,
											   pix_T_cam,None)
		if not only_return:
			self.summ_rgb(name, boxes_vis)
		return boxes_vis

	def summ_box_by_corners_parses(self, name, rgbR, corners, scores, tids, pix_T_cam,info_text, only_return=False):
		# rgb is B x H x W x C
		# corners is B x N x 8 x 3 
		# scores is B x N
		# tids is B x N
		# pix_T_cam is B x 4 x 4
		# st()
		B, C, H, W = list(rgbR.shape)
		boxes_vis = self.draw_corners_on_image(rgbR,
											   corners,
											   scores,
											   tids,
											   pix_T_cam,info_text)
		if not only_return:
			self.summ_rgb(name, boxes_vis)
		return boxes_vis

	def summ_best_orientation(self,name,val,mbr16,mbr_unpr):
		if self.save_this:
			posPair,[topkImg,topkD,topkH,topkW,topkR],ranks,[unpRs_e,unpRs_g,embs_e,embs_g],current_index = val
			
			query_index = posPair[current_index][0]
			e_index = 0
			all_targets = []
			all_best_rotations = []
			for g_index in range(1,7):
				chosen_patch_e = topkImg[int(query_index),int(e_index)]
				chosen_patch_g = topkImg[int(query_index),int(g_index)]

				pool_e_index = ranks[int(query_index),int(chosen_patch_e)]
				pool_g_index = ranks[int(query_index),int(chosen_patch_g)]
				# st()

				# torch.Size([3, 32, 32])
				unpR_e = unpRs_e[pool_e_index]
				unpR_g = unpRs_g[pool_g_index]
				# st()
				
				#torch.Size([32, 16, 16, 16])
				emb_e = embs_e[pool_e_index]
				emb_g = embs_g[pool_g_index]



				# torch.Size([1, 3, 36, 32, 32])
				unpR_g_rot = mbr_unpr.rotate2D(unpR_g.unsqueeze(0))
				# torch.Size([1, 36, 32, 16, 16, 16])
				emb_g_rot = mbr16.rotateTensor(emb_g.unsqueeze(0))
				B, angles, C, D, H, W = emb_g_rot.shape
				
				emb_g_rot = emb_g_rot.reshape(angles, -1)
				emb_e_flat = emb_e.reshape(1, -1).T
				dotprod = emb_g_rot @ emb_e_flat
				emb_g_norm = torch.norm(emb_g_rot, dim=1)
				dotprod = dotprod.T
				dotprod = dotprod/(emb_g_norm+1e-5)
				rotidx = dotprod.max(1)[1]

				beest_rot_unpRg = unpR_g_rot[0,:,rotidx[0]]
				all_best_rotations.append(beest_rot_unpRg.unsqueeze(0))
				all_targets.append(unpR_g.unsqueeze(0))
			all_best_rotations_tensors = torch.cat(all_best_rotations,dim=-1)
			all_targets_tensors = torch.cat(all_targets,dim=-1)
			target_best_tensors = torch.cat([all_targets_tensors,all_best_rotations_tensors],dim=2)
			unpR_es = torch.cat([unpR_e.unsqueeze(0),unpR_e.unsqueeze(0)],dim=2)
			unpR_es = torch.cat([unpR_es,target_best_tensors],dim=3)
			img = unpR_es
			ns = "rotation/"

			self.summ_rgb(ns+name, img)
	

	def summ_diff_orientation(self,name,pool_e, pool_g,scores,mbr16,mbr_unpr):
		if self.save_this:
			# posPair,[topkImg,topkD,topkH,topkW,topkR],ranks,[unpRs_e,unpRs_g,embs_e,embs_g],current_index = val
			embs_e, unpRs_e, _,_, visual2D_es = pool_e.fetch()
			embs_g, unpRs_g, _,_, visual2D_gs = pool_g.fetch()
			# _, unps_g, classes_g, _ ,vis2Ds_g = pool_e.fetch()			
			pool_g_index = 0
			scores_0 = scores[pool_g_index]
			pool_e_index_desc = torch.argsort(scores_0,descending=True) 
			# scores is gxe
			all_targets = []
			all_best_rotations = []
			all_vis = []
			for rank_val in range(1,7):
				# st()
				pool_e_index  = pool_e_index_desc[rank_val]
				# torch.Size([3, 32, 32])
				unpR_e = torch.from_numpy(unpRs_e[pool_e_index]).cuda().permute(2,0,1)
				visual2D_e = visual2D_es[pool_e_index].cuda()

				unpR_g = torch.from_numpy(unpRs_g[pool_g_index]).cuda().permute(2,0,1)
				visual2D_g = visual2D_gs[pool_g_index].cuda()
				
				#torch.Size([32, 16, 16, 16])
				emb_e = embs_e[pool_e_index]
				emb_g = embs_g[pool_g_index]

				# torch.Size([1, 3, 36, 32, 32])
				# st()
				unpR_e_rot = mbr_unpr.rotate2D(unpR_e.unsqueeze(0))
				# torch.Size([1, 36, 32, 16, 16, 16])
				emb_e_rot = mbr16.rotateTensor(emb_e.unsqueeze(0))
				B, angles, C, D, H, W = emb_e_rot.shape
				
				emb_e_rot = emb_e_rot.reshape(angles, -1)
				emb_g_flat = emb_g.reshape(1, -1).T
				dotprod = emb_e_rot @ emb_g_flat
				emb_e_norm = torch.norm(emb_e_rot, dim=1)
				dotprod = dotprod.T
				dotprod = dotprod/(emb_e_norm+1e-5)
				rotidx = dotprod.max(1)[1]
				beest_rot_unpRe = unpR_e_rot[0,:,rotidx[0]]
				all_best_rotations.append(beest_rot_unpRe.unsqueeze(0))
				all_targets.append(unpR_e.unsqueeze(0))
				all_vis.append(visual2D_e.unsqueeze(0))
			all_best_rotations_tensors = torch.cat(all_best_rotations,dim=-1)
			all_targets_tensors = torch.cat(all_targets,dim=-1)
			all_vis_tensors = torch.cat(all_vis,dim=-1)
			all_vis_tensors = preprocess_color(all_vis_tensors)
			visual2D_g = preprocess_color(visual2D_g)
			target_best_tensors = torch.cat([all_targets_tensors,all_best_rotations_tensors,all_vis_tensors],dim=2)
			unpR_es = torch.cat([unpR_g.unsqueeze(0),unpR_g.unsqueeze(0),visual2D_g.unsqueeze(0)],dim=2)
			unpR_es = torch.cat([unpR_es,target_best_tensors],dim=3)
			img = unpR_es
			ns = "rotation/"
			self.summ_rgb(ns+name, img)


	def summ_lrtlist(self, name, rgbR, lrtlist, scorelist, tidlist, pix_T_cam, only_return=False):
		# rgb is B x H x W x C
		# lrtlist is B x N x 17
		# scorelist is B x N
		# tidlist is B x N
		# pix_T_cam is B x 4 x 4
		
		B, C, H, W = list(rgbR.shape)
		B, N, D = list(lrtlist.shape)
		lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
		rtlist = lrtlist[:,:,3:].reshape(B, N, 4, 4)

		xyzlist_obj = utils_geom.get_xyzlist_from_lenlist(lenlist)
		# this is B x N x 8 x 3
		
		rtlist_ = rtlist.reshape(B*N, 4, 4)
		xyzlist_obj_ = xyzlist_obj.reshape(B*N, 8, 3)
		xyzlist_cam_ = utils_geom.apply_4x4(rtlist_, xyzlist_obj_)
		xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)

		boxes_vis = self.draw_corners_on_image(rgbR,
											   xyzlist_cam,
											   scorelist,
											   tidlist,
											   pix_T_cam)
		if not only_return:
			self.summ_rgb(name, boxes_vis)
		return boxes_vis
	

	def draw_boxes3D_mem_on_mem(self,img, boxes3D, scores, tids,text=False):
		# boxes3D are already in mem coords
		
		# boxes3D = trim_gt_boxes(gt_boxes3D)
		B, C, H, W = list(img.shape)
		assert(C==3)
		_, N, D = list(boxes3D.shape)
		assert(D==9)
		img = back2color(img)
		
		corners = utils_geom.transform_boxes_to_corners(boxes3D)
		corners_x = corners[:,:,:,0]
		corners_z = corners[:,:,:,2]
		# this is confusing but we stack z then x
		corners_pix = torch.stack([corners_x,corners_z], axis=3)
		# corners_pix = corners
		out = self.draw_boxes_on_image_py(img[0].detach().cpu().numpy(), corners_pix[0].detach().cpu().numpy(), scores[0].detach().cpu().numpy(), tids[0].detach().cpu().numpy(),text=text)
		# out = tf.py_func(draw_boxes3D_on_image_py, [img[0], corners_pix[0], scores[0], tids[0]], tf.uint8)
		out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
		out = torch.unsqueeze(out, dim=0)
		out = preprocess_color(out)
		out = torch.reshape(out, [1, C, H, W])		
		return out



	def draw_corners_on_image(self, rgb, corners_cam, scores, tids, pix_T_cam,info_text=None):
		# first we need to get rid of invalid gt boxes
		# gt_boxes = trim_gt_boxes(gt_boxes)
		B, C, H, W = list(rgb.shape)
		assert(C==3)
		B2, N, D, E = list(corners_cam.shape)
		assert(B2==B)
		assert(D==8) # 8 corners
		assert(E==3) # 3D

		rgb = back2color(rgb)
		corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
		corners_pix_ = utils_geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
		corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
		out = self.draw_boxes_on_image_py(rgb[0].cpu().numpy(),
										  corners_pix[0].cpu().numpy(),
										  scores[0].cpu().numpy(),
										  tids[0].cpu().numpy(),info_text)
		out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
		out = torch.unsqueeze(out, dim=0)
		out = preprocess_color(out)
		out = torch.reshape(out, [1, C, H, W])
		return out

	
	def draw_boxes_on_image_py(self, rgb, corners_pix, scores, tids,info_text=None, boxes=None, thickness=1,text=False):
		# all inputs are numpy tensors
		# rgb is H x W x 3
		# corners_pix is N x 8 x 2, in xy order
		# scores is N
		# tids is N
		# boxes is N x 9 < this is only here to print some rotation info
		# pix_T_cam is 4 x 4
		rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
		rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

		H, W, C = rgb.shape
		assert(C==3)
		N, D, E = corners_pix.shape
		assert(D==8)
		assert(E==2)

		if boxes is not None:
			rx = boxes[:,6]
			ry = boxes[:,7]
			rz = boxes[:,8]
		else:
			rx = 0
			ry = 0
			rz = 0

		color_map = matplotlib.cm.get_cmap('tab20')
		color_map = color_map.colors

		# draw
		for ind, corners in enumerate(corners_pix):
			# corners is 8 x 2
			# st()
			if not np.isclose(scores[ind], 0.0):
				# print 'score = %.2f' % scores[ind]
				color_id = tids[ind] % 20
				color = color_map[2]
				color_text = color_map[2]

				# st()

				color = np.array(color)*255.0
				# print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])
				if info_text is not None:
					text_to_put = info_text[ind]
					cv2.putText(rgb,
								text_to_put, 
								(np.min(corners[:,0]), np.min(corners[:,1])),
								cv2.FONT_HERSHEY_SIMPLEX,
								0.5, # font size
								color_text,
								2) # font weight

				for c in corners:

					# rgb[pt1[0], pt1[1], :] = 255
					# rgb[pt2[0], pt2[1], :] = 255
					# rgb[np.clip(int(c[0]), 0, W), int(c[1]), :] = 255

					c0 = np.clip(int(c[0]), 0,  W-1)
					c1 = np.clip(int(c[1]), 0,  H-1)
					rgb[c1, c0, :] = 255

				# we want to distinguish between in-plane edges and out-of-plane ones
				# so let's recall how the corners are ordered:
				xs = np.array([-1/2., -1/2., -1/2., -1/2., 1/2., 1/2., 1/2., 1/2.])
				ys = np.array([-1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2., 1/2.])
				zs = np.array([-1/2., 1/2., -1/2., 1/2., -1/2., 1/2., -1/2., 1/2.])
				xs = np.reshape(xs, [8, 1])
				ys = np.reshape(ys, [8, 1])
				zs = np.reshape(zs, [8, 1])
				offsets = np.concatenate([xs, ys, zs], axis=1)

				corner_inds = list(range(8))
				combos = list(combinations(corner_inds, 2))

				for combo in combos:
					pt1 = offsets[combo[0]]
					pt2 = offsets[combo[1]]
					# draw this if it is an in-plane edge
					eqs = pt1==pt2
					if np.sum(eqs)==2:
						i, j = combo
						pt1 = (corners[i, 0], corners[i, 1])
						pt2 = (corners[j, 0], corners[j, 1])
						retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
						if retval:
							cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)

						# rgb[pt1[0], pt1[1], :] = 255
						# rgb[pt2[0], pt2[1], :] = 255
		rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
		# utils_basic.print_stats_py('rgb_uint8', rgb)
		# imageio.imwrite('boxes_rgb.png', rgb)
		return rgb

	def summ_histogram(self, name, data):
		if self.save_this:
			data = data.flatten() 
			self.writer.add_histogram(name, data, global_step=self.global_step)

	# def summ_embeddings(self, name, emb, labels):
	# 	if not self.save_this:
	# 		return
	# 	labels = np.array(list(map(str,labels)))
	# 	self.writer.add_embedding(mat=emb,
	# 							  metadata=labels,
	# 							  global_step=self.global_step,
	# 							  tag=name)

	def flow2color(self, flow, clip=50.0):
		"""
		:param flow: Optical flow tensor.
		:return: RGB image normalized between 0 and 1.
		"""

		# flow is B x C x H x W

		B, C, H, W = list(flow.size())
		
		abs_image = torch.abs(flow)
		flow_mean = abs_image.mean(dim=[1,2,3])
		flow_std = abs_image.std(dim=[1,2,3])

		if clip:
			mf = clip
			flow = torch.clamp(flow, -mf, mf)/mf
		else:
			# Apply some kind of normalization. Divide by the perceived maximum (mean + std)
			flow = flow / (flow_mean + flow_std + 1e-10)[:, None, None, None].repeat(1, C, H, W)

		radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) #B x 1 x H x W
		radius_clipped = torch.clamp(radius, 0.0, 1.0)

		angle = torch.atan2(flow[:, 1:], flow[:, 0:1]) / np.pi #B x 1 x H x W

		hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
		saturation = torch.ones_like(hue) * 0.75
		value = radius_clipped
		hsv = torch.cat([hue, saturation, value], dim=1) #B x 3 x H x W

		#flow = tf.image.hsv_to_rgb(hsv)
		flow = hsv_to_rgb(hsv)
		flow = (flow*255.0).type(torch.ByteTensor)
		return flow

	def summ_flow(self, name, im, clip=0.0, is3D=False):
		# flow is B x C x D x W
		if self.save_this:
			# if is3D:
			#     im = im.transpose(2, 3)

			self.summ_rgb(name, self.flow2color(im, clip=clip))

	def summ_3D_flow(self, name, flow, clip=0.0):
		if self.save_this:
			self.summ_histogram('%s_flow_x' % name, flow[:,0])
			self.summ_histogram('%s_flow_y' % name, flow[:,1])
			self.summ_histogram('%s_flow_z' % name, flow[:,2])

			# flow is B x 3 x D x H x W; inside the 3 it's XYZ
			# D->z, H->y, W->x
			flow_xz = torch.cat([flow[:, 0:1], flow[:, 2:]], dim=1) # grab x, z
			flow_xy = torch.cat([flow[:, 0:1], flow[:, 1:2]], dim=1) # grab x, y
			flow_yz = torch.cat([flow[:, 1:2], flow[:, 2:]], dim=1) # grab y, z
			# these are B x 2 x D x H x W

			flow_xz = torch.mean(flow_xz, dim=3) # reduce over H (y)
			flow_xy = torch.mean(flow_xy, dim=2) # reduce over D (z)
			flow_yz = torch.mean(flow_yz, dim=4) # reduce over W (x)

			self.summ_flow('%s_flow_xz' % name, flow_xz, clip=clip, is3D=True) # rot90 for interp
			self.summ_flow('%s_flow_xy' % name, flow_xy, clip=clip)
			# self.summ_flow('%s_flow_yz' % name, flow_yz, clip=clip) # not as interpretable
			
			# flow_mag = torch.mean(torch.sum(torch.sqrt(EPS+flow**2), dim=1, keepdim=True), dim=3)
			# self.summ_oned('%s_flow_mag' % name, flow_mag, is3D=True)

	def summ_embeddings(self, name, emb, labels=None):
		if not self.save_this:
			return
		emb = emb.cpu().detach().numpy()
		N,D = emb.shape
		if labels is None:
			labels = np.arange(N)
		else:
			labels = labels.cpu().detach().numpy()
		assert labels.shape == (N,)
		labels = list(map(str,labels))
		self.writer.add_embedding(mat=emb,
					  metadata=labels,
					  global_step=self.global_step,
					  tag=name)

if __name__ == "__main__":
	logdir = './runs/my_test'
	writer = SummaryWriter(logdir = logdir)

	summ_writer = Summ_writer(writer, 0, 'my_test')

	'''test summ_rgbs'''
	# rand_input = torch.rand(1, 2, 128, 384, 3) - 0.5 #rand from -0.5 to 0.5
	# summ_rgbs(name = 'rgb', ims = torch.unbind(rand_input, dim=1), writer=writer, global_step=0)
	# rand_input = torch.rand(1, 2, 128, 384, 3) - 0.5 #rand from -0.5 to 0.5
	# summ_rgbs(name = 'rgb', ims = torch.unbind(rand_input, dim=1), writer=writer, global_step=1)

	'''test summ_occs'''
	# rand_input = torch.randint(low=0, high = 2, size=(1, 2, 32, 32, 32, 1)).type(torch.FloatTensor) #random 0 or 1
	# summ_occs(name='occs', occs=torch.unbind(rand_input, dim=1), writer=writer, global_step=0)
	# rand_input = torch.randint(low=0, high = 2, size=(1, 2, 32, 32, 32, 1)).type(torch.FloatTensor) #random 0 or 1
	# summ_occs(name='occs', occs=torch.unbind(rand_input, dim=1), writer=writer, global_step=1)

	'''test summ_unps'''
	# for global_step in [0, 1]:
	#     rand_occs = torch.randint(low=0, high = 2, size=(1, 2, 128, 128, 32, 1)).type(torch.FloatTensor) #random 0 or 1
	#     rand_unps = torch.rand(1, 2, 128, 128, 32, 3) - 0.5
	#     summ_unps(name='unps', unps=torch.unbind(rand_unps, dim=1), occs=torch.unbind(rand_occs, dim=1), writer=writer, global_step=global_step)

	'''test summ_feats'''
	# for global_step in [0, 1]:
	#     rand_feats = torch.rand(1, 2, 128, 128, 32, 3) - 0.5
	#     summ_feats(name='feats', feats=torch.unbind(rand_feats, dim=1), writer=writer, global_step=global_step)

	'''test summ_flow'''
	# rand_feats = torch.rand(2, 2, 128, 128) - 0.5
	# summ_writer.summ_flow('flow', rand_feats)

	'''test summ_flow'''
	rand_feats = torch.rand(2, 3, 128, 32, 128)
	summ_writer.summ_3D_flow(rand_feats)


	writer.close()

