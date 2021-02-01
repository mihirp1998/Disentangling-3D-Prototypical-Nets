import cv2
import os
import sys
import random
import matplotlib as mpl
import numpy as np
import random
import socket
from os.path import join, isdir
import scipy
from scipy.misc import imread
import pickle
import multiprocess_flag
multiprocess_flag.multi = True
import utils_geom
import imageio
import sys
import math
import sys
import pickle
import torch
import utils_pyplot_vis
# import utils_geom
# import tensorflow as tf
from utils_basic import *

# import cpu_gpu
# cpu_gpu.gpu = False
# from utils.tfutil import _bytes_feature
import pathos.pools as pp
import os
# st()
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import torch.nn as nn
from os import listdir
# import utils
import sys
from matplotlib import pyplot
print(sys.path)
import glob
# import utils
# print(utils.__path__)
from itertools import permutations
import random
sync_dict_keys = [
	'colorIndex1', 'colorIndex2', 'colorIndex3', 'colorIndex4', 'colorIndex5', 'colorIndex6', \
	'depthIndex1', 'depthIndex2', 'depthIndex3', 'depthIndex4', 'depthIndex5', 'depthIndex6', \
	]

EPS = 1e-6
FOCAL_LENGTH = 2.2998
empty_table = True
# NUM_VIEWS = 36

XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX = (-8.0, 8.0, -8.0, 8.0, -8.0, 8.0)


H = 256.0
W = 256.0
MAX_DEPTH_PTS = H*W
VISUALIZE = False
empty_table = False

# camX_T_R_ means it is packed(B*S,4,4)
# camX_T_R means it is unpacked(B,S,4,4)
# _p to pack

RADIUS = 13
SCENE_SIZE = 8.0
FOV = 47.0
DO_TREE= True

mod_name = sys.argv[1]
# NPY_MOD = "ac" #CLEVR_DATASET_DEFAULT_256_C with emtpy_table with tree
# # NPY_MOD = "ad" #CLEVR_DATASET_DEFAULT_256_D with emtpy_table with tree
# # NPY_MOD = "aa" #CLEVR_DATASET_DEFAULT_256_A with emtpy_table with tree
# NPY_MOD = "ab" #CLEVR_DATASET_DEFAULT_256_B with emtpy_table with tree
# NPY_MOD = "ab" # mix of aa,ab,ac,ad

# mod
# more object cases
# folderMod_dict = {"ba":"CLEVR_TEST_NEW_OBS_256_1","bb":"CLEVR_TEST_NEW_OBS_256_2"\
# ,"bc":"CLEVR_TEST_NEW_OBS_256_3","bd":"CLEVR_TEST_NEW_OBS_256_4"\
# ,"be":"CLEVR_TEST_NEW_OBS_256_5","bf":"CLEVR_TEST_NEW_OBS_256_6"\
# }

# folderMod_dict = {"ca":"CLEVR_SINGLE_LARGE_OBJ_256_A"}
# old
# folderMod_dict = {"ca":"CLEVR_SINGLE_LARGE_OBJ_256_A","cb":"CLEVR_SINGLE_LARGE_OBJ_256_B",\
# "cc":"CLEVR_SINGLE_LARGE_OBJ_256_C","cd":"CLEVR_SINGLE_LARGE_OBJ_256_D",
# "ce":"CLEVR_SINGLE_LARGE_OBJ_256_E","cf":"CLEVR_SINGLE_LARGE_OBJ_256_F",
# "da":"CLEVR_SINGLE_RANDOM_SHEAR_SCALE_OBJ_256_A","db":"CLEVR_SINGLE_RANDOM_SHEAR_SCALE_OBJ_256_B",
# "dc":"CLEVR_SINGLE_RANDOM_SHEAR_SCALE_OBJ_256_C","dd":"CLEVR_SINGLE_RANDOM_SHEAR_SCALE_OBJ_256_D",
# "de":"CLEVR_SINGLE_SHAPE_RANDOM_SHEAR_SCALE_OBJ_256_A","df":"CLEVR_SINGLE_SHAPE_RANDOM_SHEAR_SCALE_OBJ_256_B",
# "dg":"CLEVR_SINGLE_SHAPE_RANDOM_SHEAR_SCALE_OBJ_256_C","dh":"CLEVR_SINGLE_SHAPE_RANDOM_SHEAR_SCALE_OBJ_256_D",
# "ra":"CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_A", "rb":"CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_B",
# "rc":"CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_C", "rd":"CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_D",
# "re":"CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_E", "rf":"CLEVR_SINGLE_LARGE_ROTATED_OBJ_256_F",
# "ma":"CLEVR_ROTATED_MULTIPLE_256_A", "mb":"CLEVR_ROTATED_MULTIPLE_256_B",
# "mc":"CLEVR_ROTATED_MULTIPLE_256_C", "md":"CLEVR_ROTATED_MULTIPLE_256_D",
# "xa":"CLEVR_256_1", "xb":"CLEVR_256_2",
# "xc":"CLEVR_256_3", "xd":"CLEVR_256_4",
# "xe":"CLEVR_256_5",
# }
# old
folderMod_dict = {
"aa":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION","ab":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_1",
"ac":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_2","ad":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_3",
"ae":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_4","af":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_5",
"ag":"CLEVR_MULTIPLE_256_NO_ROTATION_NO_SHEAR_LOW_ELEVATION_6","ba":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_1","bb":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_2",
"bc":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_3","bd":"CLEVR_MULTIPLE_256_NO_SHEAR_LOW_ELEVATION_4",
}



NPY_MOD = mod_name 
dir_name = folderMod_dict[mod_name]


MIN_DEPTH_RANGE = RADIUS- SCENE_SIZE
MAX_DEPTH_RANGE = RADIUS + SCENE_SIZE

baxter = False
hostname = socket.gethostname()

if "MBP" in hostname:
	base_dir = "/Users/mihirprabhudesai/Documents/dataset/clevr_veggies"
	mpl.use('tkAgg')
elif "Alien" in hostname:
	base_dir = "/media/mihir/dataset/clevr_veggies"
	mpl.use('tkAgg')
elif 'baxterstation' in hostname:
	base_dir = "/home/nel/Documents/datasets/katefgroup/datasets/clevr_veggies"	
	baxter = True
	import Imath, OpenEXR
else:
	base_dir = "/projects/katefgroup/datasets/clevr_veggies/"
	base_dir = "/projects/katefgroup/datasets/clevr_vqa/output"


empty_dir_name = "CLEVR_DATASET_DEFAULT_256_EMPTY"
empty_dir = "CLEVR_new_000000"

data_dir = "{}/{}/images/train".format(base_dir,dir_name)
empty_data_dir = "{}/{}/images/train".format(base_dir,empty_dir_name)

out_dir_base = '{}/npys'.format(base_dir)
out_dir = '%s/%s' % (out_dir_base, NPY_MOD)

mkdir(out_dir)
mkdir(data_dir.replace("images","trees_updated"))

listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))

PHIS = list(range(20, 80, 20))
# PHIS = list([40])
THETAS = list(range(0, 360, 45))
all_cameras = []

for phi in PHIS:
	for theta in THETAS:
		all_cameras.append("{}_{}".format(theta,phi))

NUM_VIEWS = len(all_cameras)
random.shuffle(all_cameras)
all_cameras = all_cameras[:]

NUM_CAMS = len(all_cameras)
# st()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_exr_to_numpy(depth_path):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(depth_path)
    dw = golden.header()['dataWindow']
    redstr = golden.channel('R', pt)
    red = np.fromstring(redstr, dtype = np.float32)
    red.shape = (int(H),int(W)) # Numpy arrays are (row, col)
    depth = red
    return depth


def process_rgbs(rgb):
	H_, W_, _ = rgb.shape
	assert (H_ == 256)  # otw i don't know what data this is
	assert (W_ == 256)  # otw i don't know what data this is
	return rgb

def process_depths(depth):
	depth = depth *100.0
	depth = depth.astype(np.float32)
	return depth
	  
def depth2xyz(depth_camXs, pix_T_cams):

	"""
	  depth_camXs: B X H X W X 1

	"""
	depth_camXs = torch.tensor(depth_camXs)
	pix_T_cams = torch.tensor(pix_T_cams)
	depth_camXs = depth_camXs.permute([0,3,1,2])

	xyz_camXs = utils_geom.depth2pointcloud_cpu(depth_camXs, pix_T_cams)
	clipped_xyz = []
	for xyz_camX in xyz_camXs:	
		xyz_camX = clip(xyz_camX)
		clipped_xyz.append(xyz_camX)
	xyz_camXs = torch.stack(clipped_xyz)
	return xyz_camXs


def process_xyz(depth, pix_T_cam):	
	H, W = depth.shape
	depth = np.reshape(depth, [1, H, W, 1])
	pix_T_cams = np.expand_dims(pix_T_cam, 0)    
	xyz_camXs = depth2xyz(depth, pix_T_cams) 
	return xyz_camXs




def get_intrinsic_matrix_np():
	intrinsic_matrix = np.eye(4, dtype=np.float32)
	intrinsic_matrix[2, 2] = float(FOCAL_LENGTH)
	intrinsic_matrix[1, 1] = float(FOCAL_LENGTH)	
	intrinsic_matrix = np.reshape(intrinsic_matrix, (1, 4, 4))
	return intrinsic_matrix


def get_focal(fov):
	W=64.0
	EPS = 1e-6
	fx = W / 2.0 * 1.0 / math.tan(fov * math.pi / 180 / 2)
	fy = fx
	focal_length = fx / (W / 2.0)
	FOCAL_LENGTH = focal_length
	FOCAL_LENGTH = focal_length
	return FOCAL_LENGTH


def get_intrinsic_matrix_np(FOCAL_LENGTH, H, W):
	intrinsic_matrix = np.eye(4, dtype=np.float32)
	intrinsic_matrix[1, 1] = float(FOCAL_LENGTH) * (H * 0.5)
	intrinsic_matrix[0, 0] = float(FOCAL_LENGTH) * (W * 0.5)   
	intrinsic_matrix[0,2] = (H * 0.5)
	intrinsic_matrix[1,2] = (W * 0.5)
	# st()
	return intrinsic_matrix


def clip(xyz_camXs_single):
	MIN_DEPTH_RANGE = 5
	MAX_DEPTH_RANGE = 21

	xyz_camXs_single =  xyz_camXs_single[xyz_camXs_single[:, 2] > MIN_DEPTH_RANGE]
	xyz_camXs_single = xyz_camXs_single[xyz_camXs_single[:, 2] < MAX_DEPTH_RANGE]
	V_current = xyz_camXs_single.shape[0]

	if V_current > MAX_DEPTH_PTS:
		xyz_camXs_single = xyz_camXs_single[torch.randperm(V_current)[:V]]
	elif V_current < MAX_DEPTH_PTS:
		zeros = torch.zeros(1,3).repeat(int(MAX_DEPTH_PTS-V_current),1)
		xyz_camXs_single = torch.cat([xyz_camXs_single,zeros],axis=0)
	return xyz_camXs_single

def get_coordinates(tree):
	bbox = tree.bbox
	z,y,x,d,h,w = bbox
	transformed_coord_1 = [(W-x),(H-y),(z)]
	z2,y2,x2 = [z+d,y+h,x+w]
	transformed_coord_2 = [(W-x2),(H-y2),(z2)]
	coords = np.stack([transformed_coord_1,transformed_coord_2])
	return coords


def get_extrensic_np(theta,phi,distance):
	theta = -theta
	phi = -phi
	sin_phi = np.sin(phi / 180 * np.pi)
	cos_phi = np.cos(phi / 180 * np.pi)
	sin_theta = np.sin(theta / 180.0 * np.pi)
	cos_theta = np.cos(theta / 180.0 * np.pi)    
	rotation_azimuth_flat = [
		cos_theta, 0.0, sin_theta,
		0.0, 1.0, 0.0,
		-sin_theta, 0.0, cos_theta
	]    
	rotation_elevation_flat = [1,0,0
		,0,cos_phi,-sin_phi
		,0.0,sin_phi,cos_phi
	]    
	f = lambda x: np.reshape(np.stack(x), (3, 3))
	rotation_azimuth = f(rotation_azimuth_flat)
	rotation_elevation = f(rotation_elevation_flat)    
	rotation_matrix = np.matmul(rotation_azimuth, rotation_elevation)
	
	displacement = np.zeros((3, 1), dtype=np.float32)
		
	displacement[2, 0] = distance
	displacement = np.matmul(rotation_matrix, displacement)    
	bottom_row = np.zeros((1, 4), dtype = np.float32)
	bottom_row[0,3] = 1.0
	bottom_row = bottom_row    
	extrinsic_matrix = np.concatenate([
		np.concatenate([rotation_matrix, -displacement], axis = 1),
		bottom_row
	], axis = 0)        
	
	return extrinsic_matrix

def get_ref_T_mem(correction=True):
    # sometimes we want the mat itself
    # note this is not a rigid transform
    
    # for interpretability, let's construct this in two steps...
    #proto = coord.proto
    MH, MW, MD = (256,256,256)

    # translation
    center_T_ref = np.eye(4, dtype=np.float32)
    center_T_ref[0,3] = -XMIN
    center_T_ref[1,3] = -YMIN
    center_T_ref[2,3] = -ZMIN

    VOX_SIZE_X = (XMAX-XMIN)/float(MW)
    VOX_SIZE_Y = (YMAX-YMIN)/float(MH)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(MD)
    
    # scaling
    mem_T_center = np.eye(4, dtype=np.float32)
    mem_T_center[0,0] = 1./VOX_SIZE_X
    mem_T_center[1,1] = 1./VOX_SIZE_Y
    mem_T_center[2,2] = 1./VOX_SIZE_Z
    
    if correction:
        mem_T_center[0,3] = 0.7
        mem_T_center[1,3] = 0.5
        mem_T_center[2,3] = -0.7
    rt = np.dot(mem_T_center, center_T_ref)
    rt = np.linalg.inv(rt)
    return rt

def enum_obj_paths(IN_DIR):
	good_paths = []
	for path in listonlydir(IN_DIR):
		stuff = listdir(path)
		good_paths.append(path)
	return good_paths

def gen_cube_coordinates(coord):
	x,y,z = coord[0,0]
	x1,y1,z1 = coord[0,1]
	cube_coords = torch.tensor([[x,y,z],[x,y1,z],[x1,y,z],[x,y,z1]])
	tree_box = np.array([x,y,z,x1,y1,z1])
	return cube_coords,tree_box

def gen_list_of_bboxes(tree,boxes= [],ref_T_mem=None):
	for i in range(0, tree.num_children):
		updated_tree,boxes = gen_list_of_bboxes(tree.children[i],boxes=boxes,ref_T_mem=ref_T_mem)
		tree.children[i] = updated_tree		
	if tree.function == "describe":
		coordinates_M = get_coordinates(tree)
		coordinates_M = np.expand_dims(coordinates_M,0).astype(np.float32)
		coordinates_R = utils_geom.apply_4x4(torch.tensor(ref_T_mem), torch.tensor(coordinates_M))
		camR_T_origin = get_camRTorigin()
		coordinates_R = utils_geom.apply_4x4(torch.tensor(camR_T_origin), torch.tensor(coordinates_R,dtype=torch.float32))
		# coords = np.squeeze(coords.cpu().numpy(),axis=0)
		cube_coordinates,tree_box = gen_cube_coordinates(coordinates_R)
		tree.bbox_origin = tree_box
		boxes.append(cube_coordinates)
	return tree,boxes

def get_camRTorigin():
	rt = torch.eye(4)
	rt[2,3]= -RADIUS
	rt = np.linalg.inv(rt)
	np.expand_dims(rt,axis=0)
	return rt


def job(data_dir):
	_,image_folder = data_dir
	current_dir = image_folder.split("/")[-1]
	print(current_dir)

	for FOV in [49.5]:
		print("%d FOV"%FOV)
		focal_length = get_focal(FOV)
		focal_length = 2.1875
		intrinsics = get_intrinsic_matrix_np(focal_length, H, W)
		pix_T_cams_ = []
		rgb_camXs_ = []
		xyz_camXs_ = []
		depths = []
		
		empty_rgb_camXs_ = []
		empty_xyz_camXs_ = []
		empty_depths = []		
		camR_T_origin_ = []
		origin_T_camXs_ = []
		for cam_name in all_cameras:
			out_fn = current_dir
			out_fn += '.p'
			out_fn = '%s/%s' % (out_dir, out_fn)		
			rgb_file = join(image_folder,"%s_%s.png"%(current_dir,cam_name))
			rgb = imread(rgb_file)
			rgb_camXs_.append(process_rgbs(rgb))
			if empty_table:
				empty_rgb_file = join(empty_data_dir,"%s" % empty_dir,"%s_%s.png"%(empty_dir,cam_name))
				empty_rgb = imread(empty_rgb_file)
				empty_rgb_camXs_.append(process_rgbs(empty_rgb))
			depth_file = rgb_file.replace("images","depth").replace("png","exr")
			if baxter:
				depth =  convert_exr_to_numpy(depth_file)
			else:
				depth = np.array(imageio.imread(depth_file, format='EXR-FI'))[:,:,0]
			
			depth = process_depths(depth)
			depths.append(depth)
			if empty_table:
				empty_depth_file = empty_rgb_file.replace("images","depth").replace("png","exr")
				if baxter:
					empty_depth = convert_exr_to_numpy(empty_depth_file)
				else:				
					empty_depth = np.array(imageio.imread(empty_depth_file, format='EXR-FI'))[:,:,0]
				empty_depth = process_depths(empty_depth)
				empty_depths.append(empty_depth)
			xyz_camXs_.append(process_xyz(depth,intrinsics).cpu().numpy())
			if empty_table:
				empty_xyz_camXs_.append(process_xyz(empty_depth,intrinsics).cpu().numpy())
			theta,phi = cam_name.split("_")
			theta =float(theta)
			phi =float(phi)
			origin_T_X =np.expand_dims(get_extrensic_np(theta,phi,RADIUS), 0)
			origin_T_camXs_.append(origin_T_X)
			pix_T_cams_.append(intrinsics)
			camR_T_origin_.append(get_camRTorigin())
		if DO_TREE:
			tree_file = "/".join(rgb_file.replace("images","trees").split("/")[:-1])+".tree"
			tree = pickle.load(open(tree_file,"rb"))
			ref_T_mem = get_ref_T_mem()
			updated_tree,boxes = gen_list_of_bboxes(tree,boxes=[],ref_T_mem=ref_T_mem)
			tree_updated_file = tree_file.replace("trees","trees_updated")
			pickle.dump(updated_tree,open(tree_updated_file,"wb"))		
			cube_coordinates = np.stack(boxes)
		pix_T_cams = np.stack(pix_T_cams_, axis=0)
		rgb_camXs = np.stack(rgb_camXs_, axis=0)
		xyz_camXs = np.stack(xyz_camXs_, axis=0)
		camR_T_origin = np.stack(camR_T_origin_,axis=0)
		origin_T_camXs = np.squeeze(np.stack(origin_T_camXs_, axis=0),axis=1).astype(np.float32)
		depths = np.stack(depths, axis=0)
		# st()
		if empty_table:
			empty_rgb_camXs = np.stack(empty_rgb_camXs_, axis=0)
			empty_xyz_camXs = np.stack(empty_xyz_camXs_, axis=0)
			empty_depths = np.stack(empty_depths, axis=0)			
		assert origin_T_camXs.shape==(NUM_VIEWS,4,4)
		assert rgb_camXs.dtype == np.uint8
		assert pix_T_cams.shape == (NUM_VIEWS,4,4)
		xyz_camXs = np.squeeze(xyz_camXs,1)
		rgb_camXs_raw = rgb_camXs
		xyz_camXs_raw = xyz_camXs
		depths_raw = depths
		origin_T_camXs_raw = origin_T_camXs
		camR_T_origin_raw = camR_T_origin 
		pix_T_cams_raw = pix_T_cams
		if empty_table:
			empty_rgb_camXs_raw = empty_rgb_camXs
			empty_xyz_camXs_raw = empty_xyz_camXs
			empty_depths_raw = empty_depths
			empty_xyz_camXs_raw = np.squeeze(empty_xyz_camXs_raw,1)

		comptype = "GZIP"
		if VISUALIZE:
			mkdir("preprocess_vis/dump_npy_vis")
			ax_points = None
			origin_T_camXs_selected = []
			for cam_num in random.sample(list(range(0, NUM_VIEWS)),5):
				pix_to_cam_current = pix_T_cams[cam_num]
				rgb_current = rgb_camXs[cam_num]
				xyz_camX_current = xyz_camXs[cam_num]
				depth_current = depths[cam_num]
				origin_T_camXs_current = origin_T_camXs[cam_num]
				origin_T_camXs_selected.append(origin_T_camXs_current)
				xyz_origin = utils_geom.apply_4x4(torch.tensor(origin_T_camXs_current,dtype=torch.float32), torch.tensor(np.expand_dims(xyz_camX_current,axis=0)))
				camR_T_origin = get_camRTorigin()
				xyz_R = utils_geom.apply_4x4(torch.tensor(camR_T_origin,dtype=torch.float32), torch.tensor(xyz_origin))

				xyz_origin = xyz_origin.cpu().numpy()
				xyz_R = xyz_R.cpu().numpy()

				if not os.path.exists('preprocess_vis/dump_npy_vis/%s_rgb_cam%d.png' % (current_dir, cam_num)):
					scipy.misc.imsave('preprocess_vis/dump_npy_vis/%s_rgb_cam%d.png' % (current_dir, cam_num),rgb_current)
					scipy.misc.imsave('preprocess_vis/dump_npy_vis/%s_depth_cam%d.png' % (current_dir, cam_num),depth_current)   
				fig, ax_points = utils_pyplot_vis.plot_pointcloud(xyz_R[0,::10], fig_id=3, ax=ax_points, xlims = [-8.0, 8.0], ylims = [-8.0,8.0], zlims=[5, 21.0],coord="xright-ydown")
				if DO_TREE:
					fig, ax_points = utils_pyplot_vis.plot_cube(cube_coordinates,fig=fig,ax=ax_points)
				# fig, ax_points = utils_pyplot_vis.plot_pointcloud(xyz_R[0], fig_id=3, ax=ax_points, xlims = [-10.0, 10.0], ylims = [-10.0, 10.0], zlims=[5, 21.0],coord="xright-ydown")
				# if DO_TREE:
				# 	fig, ax_points = utils_pyplot_vis.plot_cube(cube_coordinates,fig=fig,ax=ax_points)
			# utils.pyplot_vis.plot_cam(tf.concat(origin_T_camXs_selected, 0), fig_id=2, xlims = [-13.0, 13.0], ylims = [-13.0, 13.0], zlims=[-13, 13.0], length=2.0)
			print(cam_num)
			pyplot.show()

		tree_filename = out_fn.split("/")[-1].replace(".p",".tree")
		tree_folder_info =  join(dir_name,"trees_updated/train",tree_filename) 
		# folder_info 
		if empty_table:
			feature = {
				'tree_seq_filename': tree_folder_info,
				'pix_T_cams_raw': pix_T_cams_raw,
				'origin_T_camXs_raw': origin_T_camXs_raw,
				'rgb_camXs_raw': rgb_camXs_raw,
				"camR_T_origin_raw": camR_T_origin_raw,
				# 'depth_camXs_raw': depths_raw,
				'xyz_camXs_raw': xyz_camXs_raw,
				'empty_rgb_camXs_raw': empty_rgb_camXs_raw,
				'empty_xyz_camXs_raw': empty_xyz_camXs_raw,
			}			
		else:
			feature = {
				'tree_seq_filename': tree_folder_info,
				'pix_T_cams_raw': pix_T_cams_raw,
				'origin_T_camXs_raw': origin_T_camXs_raw,
				'rgb_camXs_raw': rgb_camXs_raw,
				"camR_T_origin_raw": camR_T_origin_raw,
				'xyz_camXs_raw': xyz_camXs_raw,
			}
		feature_np = feature
		shape_dict = print_feature_shapes(feature)

		pickle.dump(feature_np,open(out_fn,"wb"))
		sys.stdout.write('.')
		sys.stdout.flush()

def print_feature_shapes(fs):
	shape_dict = {}
	for k,i in fs.items():
		if isinstance(i,type(np.array([]))):
			shape_dict[k] = i.shape
	print(shape_dict)
	return shape_dict

def main(mt):
	all_paths = enum_obj_paths(data_dir)
	all_paths.sort()
	# leaving last one always as it could be incomplete
	all_paths = all_paths[:-1]
	if mt:
		p = pp.ProcessPool(4)
		jobs = sorted(list(enumerate(all_paths)))
		print(jobs)
		p.map(job, jobs, chunksize = 1)
	else:
		for x in enumerate(all_paths):
			job(x)
	random.shuffle(all_paths)
	split = int(len(all_paths)*0.8)
	with open(out_dir_base + f'/{NPY_MOD}t.txt', 'w') as f:
		for item in all_paths[:split]:
			f.write("%s/%s\n" % (NPY_MOD,os.path.basename(item) + ".p"))

	with open(out_dir_base + f'/{NPY_MOD}v.txt', 'w') as f:
		for item in all_paths[split:]:
			f.write("%s/%s\n" % (NPY_MOD,os.path.basename(item) + ".p"))

if __name__ == '__main__':
	main(False)