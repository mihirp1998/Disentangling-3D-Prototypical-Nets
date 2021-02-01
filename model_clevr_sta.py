import torch
import torch.nn as nn
import hyperparams as hyp
import cross_corr
import numpy as np
import imageio
import os
import json
from DoublePool import ClusterPool
from model_base import Model
from nets.smoothnet import SmoothNet
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.preoccnet import PreoccNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet
from nets.embnet2D import EmbNet2D, EmbNet2D_Encoder
from nets.pixor import PIXOR as PIXOR3D
from nets.pixor import Decoder as Decoder3D
from nets.munitnet import MunitNet,MunitNet_Simple
from nets.detnet import DetNet
from nets.embnet3D import EmbNet3D
from collections import defaultdict
import torch.nn.functional as F
from scipy.misc import imsave
from collections import defaultdict
from os.path import join
import time
import random
import glob
# from utils_basic import *
from nets import pixor
import pickle
import utils_vox
import utils_samp
import utils_geom
import utils_improc
import utils_basic
import socket
import cross_corr
import utils_basic
import ipdb
st = ipdb.set_trace
import scipy
import utils_vox
# st()
import utils_eval
from archs.vector_quantizer import VectorQuantizer,VectorQuantizer_vox,VectorQuantizer_Eval,VectorQuantizer_Instance_Vr,VectorQuantizer_Instance_Vr_All,VectorQuantizer_Supervised,VectorQuantizer_Supervised_Evaluator
from archs.vector_quantizer_ema import VectorQuantizerEMA
# from arch. import VectorQuantize
import sklearn

from DoublePool import SinglePool
import torchvision.models as models

from lib_classes import Nel_Utils as nlu
import copy
import lib_classes.PointCloudBoundingBoxGenerator as PointCloudHelper
np.set_printoptions(precision=2)
np.random.seed(0)

from sklearn.cluster import MiniBatchKMeans

class CLEVR_STA(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        # self.model = CarlaStaNet().to(self.device)
        self.model = ClevrStaNet()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)
        if hyp.moc:
            self.model_key = ClevrStaNet()

class ClevrStaNet(nn.Module):
    def __init__(self):
        super(ClevrStaNet, self).__init__()
        self.device = "cuda"
        self.list_of_classes = []
        if hyp.imgnet:
            image_modules = list(list(models.resnet18().children())[:5])
            self.imagenet = nn.Sequential(*image_modules)
            self.imagenet.cuda()
            for param in self.imagenet.parameters():
                param.requires_grad = False
        
        if hyp.do_det:
            self.detnet = DetNet()
            if hyp.self_improve_once or hyp.filter_boxes:
                self.detnet_target = DetNet()
                self.detnet_target.eval()
        
        if hyp.dataset_name == "clevr":
            self.minclasses = 20
            self.minclasses = 3
        elif hyp.dataset_name == "clevr_vqa":
            self.minclasses = 3            
        elif hyp.dataset_name == "carla":
            self.minclasses = 26
        elif hyp.dataset_name == "replica":
            self.minclasses = 26            
        else:
            self.minclasses = 41
        # st()
            # confirm
        if hyp.save_embed_tsne:
            self.cluster_pool = ClusterPool(hyp.offline_cluster_pool_size)

        if hyp.aug_det:
            if hyp.aug_object_ent_dis:
                self.list_aug_content = []
                self.list_aug_style = []
                self.list_aug_classes_content = []
                self.list_aug_classes_style = []
                self.list_aug_shapes = defaultdict(lambda:[])
                self.dict_aug = {}
                self.list_aug = []
                self.list_aug_classes = []                
            elif hyp.aug_object_dis:
                self.list_aug_content = []
                self.list_aug_style = []
                self.list_aug_classes_content = []
                self.list_aug_classes_style = []
                self.list_aug_shapes = defaultdict(lambda:[])
                self.dict_aug = {}
            else:
                self.list_aug = []
                self.list_aug_classes = []
        if hyp.create_prototypes:
            self.create_protos = create_prototypes()
        
        if hyp.learn_linear_embeddings:
            self.supervised_protos = supervised_prototypes().cuda()

        if hyp.quant_init != "":
             hyp.object_quantize_init = None

        if hyp.rotate_aug:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE)

        if hyp.object_quantize or hyp.filter_boxes or hyp.self_improve_iterate:
            embed_size = hyp.BOX_SIZE*hyp.BOX_SIZE*hyp.BOX_SIZE*hyp.feat_dim
            if hyp.object_ema:
                self.quantizer = VectorQuantizerEMA(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)
            elif hyp.use_instances_variation:
                self.quantizer = VectorQuantizer_Instance_Vr(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)                
            elif hyp.use_instances_variation_all:
                self.quantizer = VectorQuantizer_Instance_Vr_All(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)            
            elif hyp.use_instances_variation_all:
                self.quantizer = VectorQuantizer(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)
            elif hyp.use_supervised:
                self.quantizer = VectorQuantizer_Supervised(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)

                self.quantizer_evaluator = VectorQuantizer_Supervised_Evaluator(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)

            else:
                self.quantizer = VectorQuantizer(num_embeddings=hyp.object_quantize_dictsize,
                                             embedding_dim=embed_size,
                                             init_embeddings=hyp.object_quantize_init,
                                             commitment_cost=hyp.object_quantize_comm_cost)
        if hyp.voxel_quantize:
            embed_size = hyp.feat_dim
            self.quantizer = VectorQuantizer_vox(num_embeddings=hyp.voxel_quantize_dictsize,
                                         embedding_dim=embed_size,
                                         init_embeddings=hyp.voxel_quantize_init,
                                         commitment_cost=hyp.voxel_quantize_comm_cost)

        if hyp.gt_rotate_combinations:
            self.mbr_unpr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE*2,hyp.BOX_SIZE*2,hyp.BOX_SIZE*2)
            # self.mbr_unpr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE*2,hyp.BOX_SIZE*2,hyp.BOX_SIZE*2)

            # self.mbr3d = cross_corr.meshgrid_based_rotation(hyp.Z2,hyp.Y2,hyp.X2)
        
        self.info_dict = defaultdict(lambda:[])
            
            
        self.embed_list_style = defaultdict(lambda:[])
        self.embed_list_content = defaultdict(lambda:[])

        if hyp.create_example_dict:
            self.embed_dict = defaultdict(lambda:0)
            self.embed_list = []
        
        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ or (hyp.remove_air and hyp.aug_det):
            self.occnet = OccNet()
        if hyp.do_preocc:
            self.preoccnet = PreoccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_render:
            self.rendernet = RenderNet()
        if hyp.do_emb2D:
            self.embnet2D = EmbNet2D()
            self.embnet2D_encoder = EmbNet2D_Encoder()
        if hyp.moc_2d:
            self.embnet2D_encoder = EmbNet2D_Encoder()                    
        if hyp.do_emb3D:
            self.embnet3D = EmbNet3D()
        
        if hyp.do_munit:
            if hyp.simple_adaingen:
                self.munitnet = MunitNet_Simple().cuda()
            else:
                self.munitnet = MunitNet().cuda()

        if hyp.do_smoothnet or hyp.smoothness_with_noloss:
            self.smoothnet = SmoothNet().cuda()


        if hyp.aug_object_dis or hyp.aug_object_ent_dis:
            self.munitnet = MunitNet_Simple().cuda()

        if hyp.online_cluster:
            self.kmeans = MiniBatchKMeans(n_clusters=hyp.object_quantize_dictsize,
                                          # init=np.load('vqvae/cluster_centers.npy'),
                                          # batch_size=int(hyp.Z*hyp.Y*hyp.X/8),
                                          verbose=True)
            self.voxel_queue = SinglePool(hyp.initial_cluster_size)
        if hyp.offline_cluster_eval:
            if hyp.use_kmeans:
                self.kmeans = pickle.load(open("offline_obj_cluster/kmeans.p","rb"))
            elif hyp.use_vq_vae:
                self.quantizer = VectorQuantizer_Eval()
        if hyp.offline_cluster:   
            if "compute" in socket.gethostname():
                self.kmeans = sklearn.cluster.KMeans(n_clusters=hyp.object_quantize_dictsize,verbose=True)
            else:
                self.kmeans = sklearn.cluster.KMeans(n_clusters=hyp.object_quantize_dictsize,verbose=True)

        if hyp.do_gt_pixor_det or hyp.do_pixor_det:
            use_bn = False

            self.geom = {
                "L1": utils_vox.ZMIN,
                "L2": utils_vox.ZMAX,
                "W1": utils_vox.XMIN,
                "W2": utils_vox.XMAX,
                "H1": utils_vox.YMIN,
                "H2": utils_vox.YMAX,
                "input_shape": [hyp.Z, hyp.Y, hyp.X, 1],
                "label_shape": [hyp.Z//2, hyp.Y//2, hyp.X//2, 7]
            }

            config = self.load_config("default")

            # st()
            self.pixor = PIXOR3D(self.geom)
            self.pixor.to(self.device)

            self.decoder_boxes = Decoder3D(self.geom)

            # self.loss_fn = CustomLoss(self.device, config, num_classes=1)

        self.bounding_box_generator = PointCloudHelper.BoundBoxGenerator()
        self.is_empty_occ_generated = False

        self.avg_ap = []
        self.avg_precision = []                    
        self.tp_style = 0
        self.all_style = 0

        self.tp_content = 0
        self.all_content = 0        
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.max_content = None
        self.min_content = None 
        self.max_style = None 
        self.min_style = None
        self.styles_prediction = defaultdict(lambda:[])
        self.content_prediction = defaultdict(lambda:[])

        if hyp.load_content_style_range_from_file:
            if hyp.normalize_contrast:
                d = pickle.load(open('content_style_range.p','rb'))
                self.max_content = torch.tensor(d['max_content']).cuda().unsqueeze(0)
                self.min_content = torch.tensor(d['min_content']).cuda().unsqueeze(0)
                self.max_style = torch.tensor(d['max_style']).cuda().unsqueeze(0)
                self.min_style = torch.tensor(d['min_style']).cuda().unsqueeze(0)

    def load_config(self,exp_name):
        path = os.path.join('experiments', exp_name, 'config.json')
        with open(path) as file:
            config = json.load(file)
        assert config['name']==exp_name
        return config


    def evaluate_filter_boxes(self,gt_boxesRMem_theta,scores,featR,summ_writer):
        if hyp.filter_boxes or hyp.self_improve_iterate:
            with torch.no_grad():
                if hyp.self_improve_iterate:
                    _, boxlist_memR_e_pre_filter, scorelist_e_pre_filter, _, _, _ = self.detnet(
                            self.axboxlist_memR,
                            self.scorelist_s,
                            featR,
                            summ_writer)                    
                else:
                    _, boxlist_memR_e_pre_filter, scorelist_e_pre_filter, _, _, _ = self.detnet_target(
                        self.axboxlist_memR,
                        self.scorelist_s,
                        featR,
                        summ_writer)
            summ_writer.summ_box_mem_on_mem('detnet/pre_filter_boxesR_mem', self.unp_visRs, boxlist_memR_e_pre_filter ,scorelist_e_pre_filter,torch.ones_like(scorelist_e_pre_filter,dtype=torch.int32))

            corners_memR_e_pred = utils_geom.transform_boxes_to_corners(boxlist_memR_e_pre_filter)
            end_memR_e_pred = nlu.get_ends_of_corner(corners_memR_e_pred)
            # end_memR_e_pred = torch.stack([torch.clamp(end_memR_e_pred[:,:,:,0],min=0,max=hyp.X2),torch.clamp(end_memR_e_pred[:,:,:,1],min=0,max=hyp.Y2),torch.clamp(end_memR_e_pred[:,:,:,2],min=0,max=hyp.Z2)],dim=-1)

            emb3D_e_R = utils_vox.apply_4x4_to_vox(self.camR_T_camX0, self.emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(self.camR_T_camX0, self.emb3D_g)

            emb3D_R = emb3D_e_R

            neg_boxesMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det,2,3]).cuda()
            neg_scoresMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det]).cuda()

            gt_boxesMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det,2,3]).cuda()
            gt_scoresMem_to_consider_after_cs  = torch.zeros([hyp.B,self.N_det])

            gt_boxesMem_to_consider_after_q_distance  = torch.zeros([hyp.B,self.N_det,2,3]).cuda()
            gt_scoresMem_to_consider_after_q_distance  = torch.zeros([hyp.B,self.N_det])


            emb3D_e_R_object, emb3D_g_R_object, indices, end_memR_e_pred_filtered, neg_indices, neg_boxes = nlu.create_object_tensors_filter_cs([emb3D_e_R, emb3D_g_R],  end_memR_e_pred, scorelist_e_pre_filter,[hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE], cs_check= hyp.cs_filter)

            feat_mask = torch.zeros([hyp.B,1,hyp.Z2,hyp.Y2,hyp.X2]).cuda()
            feat_mask_vis = torch.ones([1,3,hyp.Z2,hyp.X2]).cuda()*-0.5

            validR_combo_object = None
            
            if emb3D_e_R_object is not None:
                if hyp.cs_filter:                            
                    for ind,index_val in enumerate(indices):
                        batch_index, box_index = index_val
                        box_val = end_memR_e_pred_filtered[ind]
                        assert  (end_memR_e_pred_filtered[ind] == end_memR_e_pred[batch_index,box_index]).all()
                        gt_scoresMem_to_consider_after_cs[batch_index,box_index] = 1.0
                        gt_boxesMem_to_consider_after_cs[batch_index,box_index] = box_val
                    
                    if neg_boxes is not None:                                    
                        for neg_ind,neg_index_val in enumerate(neg_indices):
                            batch_index, box_index = neg_index_val
                            neg_box_val = neg_boxes[neg_ind]
                            assert  (neg_boxes[neg_ind] == end_memR_e_pred[batch_index,box_index]).all()
                            neg_boxesMem_to_consider_after_cs[batch_index,box_index] = neg_box_val
                            neg_scoresMem_to_consider_after_cs[batch_index,box_index] = 1.0

                    gt_boxesMem_to_consider_after_cs_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_cs)
                    summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_cs', self.unp_visRs, gt_boxesMem_to_consider_after_cs_theta ,gt_scoresMem_to_consider_after_cs,torch.ones([hyp.B,6],dtype=torch.int32))
                    neg_boxesMem_to_consider_after_cs_theta = nlu.get_alignedboxes2thetaformat(neg_boxesMem_to_consider_after_cs)
                    summ_writer.summ_box_mem_on_mem('detnet/sudo_neg_mem_filtered_cs', self.unp_visRs, neg_boxesMem_to_consider_after_cs_theta ,neg_scoresMem_to_consider_after_cs,torch.ones([hyp.B,6],dtype=torch.int32))


                emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2

                emb3D_R_object.shape[0] == indices.shape[0]

                distances = self.quantizer(emb3D_R_object)
                min_distances = torch.min(distances,dim=1).values


                selections = 0
                for i in range(distances.shape[0]):
                    min_distance = min_distances[i]
                    if min_distance <hyp.dict_distance_thresh:
                        selections += 1
                        index_val = indices[i]
                        batch_index, box_index = index_val
                        box_val = end_memR_e_pred_filtered[i]
                        gt_scoresMem_to_consider_after_q_distance[batch_index,box_index] = 1.0
                        gt_boxesMem_to_consider_after_q_distance[batch_index,box_index] = box_val


                if selections > 0:
                    gt_boxesMem_to_consider_after_q_distance = torch.stack([torch.clamp(gt_boxesMem_to_consider_after_q_distance[:,:,:,0],min=0,max=hyp.X2),torch.clamp(gt_boxesMem_to_consider_after_q_distance[:,:,:,1],min=0,max=hyp.Y2),torch.clamp(gt_boxesMem_to_consider_after_q_distance[:,:,:,2],min=0,max=hyp.Z2)],dim=-1)
                    for b_index in range(hyp.B):
                        for n_index in range(self.N_det):
                            if gt_scoresMem_to_consider_after_q_distance[b_index,n_index] > 0.0:
                                box = gt_boxesMem_to_consider_after_q_distance[b_index,n_index]
                                lower,upper = torch.unbind(box)

                                xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
                                xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]
                                assert (xmax-xmin) >0 and (ymax-ymin) >0 and (zmax-zmin) >0

                                padding = 3
                                xmin_padded,ymin_padded,zmin_padded = (max(xmin-padding,0),max(ymin-padding,0),max(zmin-padding,0))
                                xmax_padded,ymax_padded,zmax_padded = (min(xmax+padding,hyp.X2),min(ymax+padding,hyp.Y2),min(zmax+padding,hyp.Z2))

                                feat_mask[b_index,:,zmin_padded:zmax_padded,ymin_padded:ymax_padded,xmin_padded:xmax_padded] = 1.0
                                if b_index == 0:
                                    feat_mask_vis[b_index,1,zmin_padded:zmax_padded,xmin_padded:xmax_padded] = 0.5
                                    feat_mask_vis[b_index,1,zmin:zmax,xmin:xmax] = 0.1



                if neg_boxes is not None:
                    neg_boxesMem_to_consider_after_cs = torch.stack([torch.clamp(neg_boxesMem_to_consider_after_cs[:,:,:,0],min=0,max=hyp.X2),torch.clamp(neg_boxesMem_to_consider_after_cs[:,:,:,1],min=0,max=hyp.Y2),torch.clamp(neg_boxesMem_to_consider_after_cs[:,:,:,2],min=0,max=hyp.Z2)],dim=-1)
                    for b_index in range(hyp.B):
                        for n_index in range(self.N_det):
                            if neg_scoresMem_to_consider_after_cs[b_index,n_index] > 0.0:
                                box = neg_boxesMem_to_consider_after_cs[b_index,n_index]
                                lower,upper = torch.unbind(box)

                                xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
                                xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]
                                assert (xmax-xmin) >0 and (ymax-ymin) >0 and (zmax-zmin) >0

                                padding = 0
                                xmin_padded,ymin_padded,zmin_padded = (max(xmin-padding,0),max(ymin-padding,0),max(zmin-padding,0))
                                xmax_padded,ymax_padded,zmax_padded = (min(xmax+padding,hyp.X2),min(ymax+padding,hyp.Y2),min(zmax+padding,hyp.Z2))

                                feat_mask[b_index,:,zmin_padded:zmax_padded,ymin_padded:ymax_padded,xmin_padded:xmax_padded] = 1.0
                                if b_index == 0:
                                    feat_mask_vis[b_index,0,zmin_padded:zmax_padded,xmin_padded:xmax_padded] = 0.1


                gt_boxesMem_to_consider_after_q_distance_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_q_distance)
                summ_writer.summ_rgb('detnet/mask_vis', feat_mask_vis)
                summ_writer.summ_occ('detnet/mask_used', feat_mask)
                summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_quant', self.unp_visRs, gt_boxesMem_to_consider_after_q_distance_theta ,gt_scoresMem_to_consider_after_q_distance,torch.ones([hyp.B,6],dtype=torch.int32))
            else:
                gt_boxesMem_to_consider_after_cs_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_cs)
                gt_boxesMem_to_consider_after_q_distance_theta = nlu.get_alignedboxes2thetaformat(gt_boxesMem_to_consider_after_q_distance)
                summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_cs', self.unp_visRs, torch.zeros_like(gt_boxesRMem_theta) ,torch.zeros([hyp.B,6]),torch.ones([hyp.B,6],dtype=torch.int32))
                summ_writer.summ_occ('detnet/mask_vis', feat_mask_vis)
                summ_writer.summ_occ('detnet/mask_used', feat_mask)            
                summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_mem_filtered_quant', self.unp_visRs, torch.zeros_like(gt_boxesRMem_theta) ,torch.zeros([hyp.B,6]),torch.ones([hyp.B,6],dtype=torch.int32))
        return gt_boxesMem_to_consider_after_q_distance_theta, gt_scoresMem_to_consider_after_q_distance, feat_mask, gt_boxesMem_to_consider_after_cs_theta,gt_scoresMem_to_consider_after_cs

    def forward(self, feed):
        results = dict()
        # st()
        if 'log_freq' not in feed.keys():
            feed['log_freq'] = None
        start_time = time.time()


        summ_writer = utils_improc.Summ_writer(writer=feed['writer'],
                                               global_step=feed['global_step'],
                                               set_name=feed['set_name'],
                                               log_freq=feed['log_freq'],
                                               fps=8)
        writer = feed['writer']
        global_step = feed['global_step']

          

        # st()
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils_basic.pack_seqdim(x, B)
        __u = lambda x: utils_basic.unpack_seqdim(x, B)

        __pb = lambda x: utils_basic.pack_boxdim(x, hyp.N)
        __ub = lambda x: utils_basic.unpack_boxdim(x, hyp.N)
        if hyp.aug_object_ent_dis:
            __pb_a = lambda x: utils_basic.pack_boxdim(x, hyp.max_obj_aug + hyp.max_obj_aug_dis)
            __ub_a = lambda x: utils_basic.unpack_boxdim(x, hyp.max_obj_aug + hyp.max_obj_aug_dis)            
        else:
            __pb_a = lambda x: utils_basic.pack_boxdim(x, hyp.max_obj_aug)
            __ub_a = lambda x: utils_basic.unpack_boxdim(x, hyp.max_obj_aug)

        B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
        PH, PW = hyp.PH, hyp.PW
        K = hyp.K
        BOX_SIZE = hyp.BOX_SIZE
        Z, Y, X = hyp.Z, hyp.Y, hyp.X
        Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
        Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
        D = 9
        
        tids = torch.from_numpy(np.reshape(np.arange(B*N),[B,N]))
        # rgb_camRs = feed["rgb_camRs"]
        rgb_camXs = feed["rgb_camXs_raw"]
        pix_T_cams = feed["pix_T_cams_raw"]
        # cam_T_velos = feed["cam_T_velos"]
        camRs_T_origin = feed["camR_T_origin_raw"]
        # st()
        # st()
        origin_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_origin)))
        origin_T_camXs = feed["origin_T_camXs_raw"]
        # st()
        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
        camX0_T_camRs = camXs_T_camRs[:,0]
        camX1_T_camRs = camXs_T_camRs[:,1]
        
        camR_T_camX0  = utils_geom.safe_inverse(camX0_T_camRs)

        xyz_camXs = feed["xyz_camXs_raw"]
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))

        if hyp.replaceRD:
            depth_camXs = __u(depth_camXs_)
            depth_norm = utils_improc.preprocess_depth(depth_camXs)
            sudo_rgb = torch.cat([depth_norm,depth_norm,depth_norm],dim=2)
            rgb_camXs = sudo_rgb

        if hyp.low_res:
            if hyp.dataset_name == "carla" or hyp.dataset_name == "carla_mix" or hyp.dataset_name == "carla_det":
                xyz_camXs = __u(dense_xyz_camXs_)

        if hyp.do_empty:
            empty_rgb_camXs = feed['empty_rgb_camXs_raw']
            empty_xyz_camXs = feed['empty_xyz_camXs_raw']
            empty_xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(empty_xyz_camXs)))
            empty_xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(empty_xyz_camXs)))
            empty_occXs = __u(utils_vox.voxelize_xyz(__p(empty_xyz_camXs), Z, Y, X))
            empty_occXs_half = __u(utils_vox.voxelize_xyz(__p(empty_xyz_camXs), Z2, Y2, X2))
            empty_occX0s_half = __u(utils_vox.voxelize_xyz(__p(empty_xyz_camX0s), Z2, Y2, X2))


            empty_unpXs = __u(utils_vox.unproject_rgb_to_mem(
                __p(empty_rgb_camXs), Z, Y, X, __p(pix_T_cams)))

            empty_unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
                __p(empty_rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))


            empty_depth_camXs_, empty_valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(empty_xyz_camXs), H, W)
            empty_dense_xyz_camXs_ = utils_geom.depth2pointcloud(empty_depth_camXs_, __p(pix_T_cams))
            empty_dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), empty_dense_xyz_camXs_)
            empty_inbound_camXs_ = utils_vox.get_inbounds(empty_dense_xyz_camRs_, Z, Y, X).float()
            empty_inbound_camXs_ = torch.reshape(empty_inbound_camXs_, [B*S, 1, H, W])
            
            empty_depth_camXs = __u(empty_depth_camXs_)
            empty_valid_camXs = __u(empty_valid_camXs_) * __u(empty_inbound_camXs_)

            summ_writer.summ_oneds('2D_inputs/empty_depth_camXs', torch.unbind(empty_depth_camXs, dim=1),maxdepth=21.0)
            summ_writer.summ_oneds('2D_inputs/empty_valid_camXs', torch.unbind(empty_valid_camXs, dim=1))
            # summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
            summ_writer.summ_rgbs('2D_inputs/empty_rgb_camXs', torch.unbind(empty_rgb_camXs, dim=1))
            summ_writer.summ_occs('3D_inputs/empty_occXs', torch.unbind(empty_occXs, dim=1))
            summ_writer.summ_unps('3D_inputs/empty_unpXs', torch.unbind(empty_unpXs, dim=1), torch.unbind(empty_occXs, dim=1))
        # xyz_camXs = __u(utils_geom.apply_4x4(__p(cam_T_velos), __p(xyz_veloXs)))
        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        

        if hyp.imgnet:
            rgb_pret = rgb_camXs.reshape([-1,3,hyp.H,hyp.W])
            rgb_pret = self.imagenet(rgb_pret)
            rgb_pret = rgb_pret.reshape([hyp.B,hyp.S,64,64,64])
            unpXs_pret = __u(utils_vox.unproject_rgb_to_mem(__p(rgb_pret), Z, Y, X, __p(pix_T_cams)))

        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        # bug only in debug
        occXs_to_Rs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, occXs) # torch.Size([2, 2, 1, 144, 144, 144])
        occXs_to_Rs_45 = cross_corr.rotate_tensor_along_y_axis(occXs_to_Rs, 45)
        # remove later

        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        # occRs_half = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, occXs_half) # torch.Size([2, 2, 1, 72, 72, 72])
        occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))
        # st()

        unpXs = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z, Y, X, __p(pix_T_cams)))

        unpXs_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, __p(pix_T_cams)))

        unpX0s_half = __u(utils_vox.unproject_rgb_to_mem(
            __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camX0_T_camXs)))))

        unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z, Y, X, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
        
        unpRs_half = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(__p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
            
         
        ## projected depth, and inbound mask
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)

        #####################
        ## visualize what we got
        #####################
        # st()
        # if hyp.do_debug:
        #     for index,rgb in enumerate(rgb_camXs):
        #         imsave(f'dump/{global_step}_{index}.png',rgb[0].permute(1,2,0).cpu())
        #     st()

        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1),maxdepth=21.0)
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        # summ_writer.summ_rgbs('2D_inputs/rgb_camRs', torch.unbind(rgb_camRs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))

        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        
        if hyp.profile_time:
            print("landmark time",time.time()-start_time)

        if summ_writer.save_this:
            summ_writer.summ_occs('3D_inputs/occRs', torch.unbind(occRs, dim=1))
            summ_writer.summ_occs('3D_inputs/occXs_to_Rs', torch.unbind(occXs_to_Rs, dim=1))
            summ_writer.summ_occs('3D_inputs/occXs_to_Rs_45', torch.unbind(occXs_to_Rs_45, dim=1))
            summ_writer.summ_unps('3D_inputs/unpRs', torch.unbind(unpRs, dim=1), torch.unbind(occRs, dim=1))



        if hyp.use_gt_occs:
            occR_complete = feed['occR_complete']
            summ_writer.summ_occ('3D_inputs/occR_complete',occR_complete)

        #####################
        ## run the nets
        #####################

        if hyp.do_preocc:
            # pre-occ is a pre-estimate of occupancy
            # as another mnemonic, it marks the voxels we will preoccupy ourselves with            
            unpRs = __u(utils_vox.unproject_rgb_to_mem(
                __p(rgb_camXs), Z2, Y2, X2, utils_basic.matmul2(
                    __p(pix_T_cams), utils_geom.safe_inverse(__p(camRs_T_camXs)))))
            occR0_sup, freeR0_sup, occRs, freeRs = utils_vox.prep_occs_supervision(
                camRs_T_camXs,
                xyz_camXs,
                Z2, Y2, X2, 
                agg=True)
            summ_writer.summ_occ('occ_sup/occR0_sup', occR0_sup)
            summ_writer.summ_occ('occ_sup/freeR0_sup', freeR0_sup)
            summ_writer.summ_occs('occ_sup/freeRs_sup', torch.unbind(freeRs, dim=1))
            summ_writer.summ_occs('occ_sup/occRs_sup', torch.unbind(occRs, dim=1))
            preoccR0_input = torch.cat([
                occRs[:,0],
                freeRs[:,0],
                occRs[:,0]*unpRs[:,0]
            ], dim=1)
            preocc_loss, compR0 = self.preoccnet(
                preoccR0_input,
                occR0_sup,
                freeR0_sup,
                summ_writer,
            )
            total_loss += preocc_loss
        start_time = time.time()
        
        if hyp.do_eval_boxes:
            if hyp.dataset_name == "carla":
                tree_seq_filename = feed['tree_seq_filename']
                tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename]
                trees = [pickle.load(open(i,"rb")) for i in tree_filenames]           
                gt_boxes_origin,scores,classes = nlu.trees_rearrange_corners(trees)
                gt_boxes_origin = torch.from_numpy(gt_boxes_origin).cuda().to(torch.float)
                if hyp.use_2d_boxes:
                    prd_boxes = feed['predicted_box']
                    prd_scores = feed['predicted_scores'].detach().cpu().numpy()
                    gt_boxes_origin = prd_boxes
                    scores = prd_scores
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                # st()
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))

                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
                rgb_camtop = feed['rgb_camtop'].squeeze(1)
                origin_T_camXs_top = feed['origin_T_camXs_top']
                # camRs_T_camXTop = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs[:,:1])), __p(origin_T_camXs_top)))
                gt_boxescamXTop_corners = __ub(utils_geom.apply_4x4(utils_geom.safe_inverse(__p(origin_T_camXs_top)), __pb(gt_boxes_origin_corners)))
                # st()
                # gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta)
            elif hyp.dataset_name =="carla_mix"  or hyp.dataset_name == "carla_det":
                predicted_box_origin = feed['predicted_box']
                predicted_scores_origin = feed['predicted_scores']
                gt_boxes_origin = feed['gt_box']
                gt_scores_origin = feed['gt_scores']
                classes = feed['classes']
                tree_seq_filename = feed['tree_seq_filename']
                scores = gt_scores_origin
                if hyp.use_2d_boxes:
                    gt_boxes_origin = predicted_box_origin
                    scores = predicted_scores_origin
                scores = scores.detach().cpu().numpy()
                # st()
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                # st()
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))
                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
            
            elif hyp.dataset_name =="clevr_vqa":
                gt_boxes_origin_corners = feed['gt_box']
                gt_scores_origin = feed['gt_scores'].detach().cpu().numpy()
                classes = feed['classes']
                scores = gt_scores_origin
                tree_seq_filename = feed['tree_seq_filename']
                # st()
                gt_boxes_origin = nlu.get_ends_of_corner(gt_boxes_origin_corners)
                # gt_boxes_origin = torch.from_numpy(gt_boxes_origin).cuda().to(torch.float)

                if hyp.use_2d_boxes:
                    prd_boxes = feed['predicted_box']
                    prd_scores = feed['predicted_scores'].detach().cpu().numpy()
                    gt_boxes_origin = prd_boxes
                    scores = prd_scores
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))
                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
                gt_boxesR_end = nlu.get_ends_of_corner(gt_boxesR_corners)
                # st()
                # gt_boxesR_end = torch.reshape(gt_boxesR,[hyp.B,hyp.N,2,3])
                # rgb_camtop = feed['rgb_camtop'].squeeze(1)
                # origin_T_camXs_top = feed['origin_T_camXs_top']
                # camRs_T_camXTop = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs[:,:1])), __p(origin_T_camXs_top)))
                # gt_boxescamXTop_corners = __ub(utils_geom.apply_4x4(utils_geom.safe_inverse(__p(origin_T_camXs_top)), __pb(gt_boxes_origin_corners)))

            elif hyp.dataset_name =="replica":
                gt_boxes_origin = feed['gt_box']
                gt_scores_origin = feed['gt_scores']
                classes = feed['classes']
                scores = gt_scores_origin
                tree_seq_filename = feed['tree_seq_filename']
                if hyp.moc or hyp.do_emb3D:
                    gt_boxes_origin_f = gt_boxes_origin[:,:1].cpu().detach().numpy()
                    gt_scores_origin_f = gt_scores_origin[:,:1].cpu().detach().numpy()
                    classes_f = classes[:,:1]                    
                    N_new = 1
                    gt_boxes_origin = torch.from_numpy(np.pad(gt_boxes_origin_f,[[0,0],[0,hyp.N-N_new],[0,0]])).cuda()
                    gt_scores_origin = torch.from_numpy(np.pad(gt_scores_origin_f,[[0,0],[0,hyp.N-N_new]])).cuda()
                    classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])                    
                    scores = gt_scores_origin
                # st()
                # if hyp.use_2d_boxes:
                #     gt_boxes_origin = predicted_box_origin
                #     scores = predicted_scores_origin
                scores = scores.detach().cpu().numpy()
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                # st()
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))

                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)                                
            else:
                tree_seq_filename = feed['tree_seq_filename']
                # st() 
                tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename if i != "invalid_tree"]
                invalid_tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename if i == "invalid_tree"]
                num_empty = len(invalid_tree_filenames)
                if num_empty > 0:
                    
                    empty_gt_boxesR = np.zeros((num_empty, hyp.N, 6))
                    empty_scores = np.zeros((num_empty, hyp.N))
                    empty_classes = np.zeros((num_empty, hyp.N)).astype('U1')
                    
                trees = [pickle.load(open(i,"rb")) for i in tree_filenames]
                # st()
                if hyp.use_det_boxes:
                    # This if condition has not been checked for empty scenes
                    # bbox_detsR = [tree.bbox_det for tree in trees]
                    bbox_detsR = []
                    score_dets = []
                    for tree in trees:
                        if not hasattr(tree, 'bbox_det'):
                            bbox_detsR.append(torch.zeros([hyp.N,9]).cuda())
                            score_dets.append(torch.zeros([6]).cuda())
                        else:
                            bbox_detsR.append(tree.bbox_det)
                            score_dets.append(tree.score_det)                    
                    bbox_detsR = torch.stack(bbox_detsR)
                    bbox_dets_cornersR = utils_geom.transform_boxes_to_corners(bbox_detsR)
                    # st()
                    bbox_dets_cornersR = __ub(utils_vox.Mem2Ref(__pb(bbox_dets_cornersR),Z2,Y2,X2))
                    bbox_dets_endR = nlu.get_ends_of_corner(bbox_dets_cornersR).cpu().detach().numpy()
                    # score_dets = [tree.score_det for tree in trees]
                    score_dets = torch.stack(score_dets).cpu().detach().numpy()
                    gt_boxesR = bbox_dets_endR
                    scores = score_dets
                    best_indices = np.flip(np.argsort(scores),axis=1)
                    sorted_scores = []
                    sorted_boxes = []

                    for ind,sorted_index in enumerate(best_indices):
                        sorted_scores.append(scores[ind][sorted_index])
                        sorted_boxes.append(gt_boxesR[ind][sorted_index])
                    sorted_scores = np.stack(sorted_scores)
                    sorted_boxes = np.stack(sorted_boxes)
                    classes = np.reshape(['temp']*hyp.N*hyp.B,[hyp.B,hyp.N])

                    # take top2
                    gt_boxesR_f = sorted_boxes[:,:2]
                    gt_scoresR_f = sorted_scores[:,:2]
                    classes_f = classes[:,:2]                    
                    N_new = gt_boxesR_f.shape[1]
                    # st()
                    gt_boxesR = np.pad(gt_boxesR_f,[[0,0],[0,hyp.N-N_new],[0,0],[0,0]])
                    scores = np.pad(gt_scoresR_f,[[0,0],[0,hyp.N-N_new]])
                    classes = np.pad(classes_f,[[0,0],[0,hyp.N-N_new]])

                elif hyp.use_2d_boxes:
                    # This if condition has not been checked for empty scenes
                    gt_boxesR,scores,classes = nlu.trees_rearrange_2d(trees)
                else:
                    len_valid = len(trees)
                    if len_valid > 0:
                        gt_boxesR,scores,classes = nlu.trees_rearrange(trees)
                    
                    if num_empty > 0:
                        gt_boxesR = np.concatenate([gt_boxesR, empty_gt_boxesR]) if len_valid>0 else empty_gt_boxesR
                        scores = np.concatenate([scores, empty_scores]) if len_valid>0 else empty_scores
                        classes = np.concatenate([classes, empty_classes]) if len_valid>0 else empty_classes

                gt_boxesR = torch.from_numpy(gt_boxesR).cuda().float() # torch.Size([2, 3, 6])
                gt_boxesR_end = torch.reshape(gt_boxesR,[hyp.B,hyp.N,2,3])
                gt_boxesR_theta = nlu.get_alignedboxes2thetaformat(gt_boxesR_end) #torch.Size([2, 3, 9])
                gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta)
            # st()
            class_names_ex_1 = "_".join(classes[0])
            summ_writer.summ_text('eval_boxes/class_names', class_names_ex_1)
            
            gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))
            gt_boxesRMem_end = nlu.get_ends_of_corner(gt_boxesRMem_corners)
            
            if hyp.dataset_name == "carla" or hyp.dataset_name == "carla_mix"  or hyp.dataset_name == "carla_det":
                gt_boxesR_end = __ub(utils_vox.Mem2Ref(__pb(gt_boxesRMem_end),Z2,Y2,X2))
                # st()
                gt_boxesR_theta = nlu.get_alignedboxes2thetaformat(gt_boxesR_end) #torch.Size([2, 3, 9])
                gt_boxesR_corners = utils_geom.transform_boxes_to_corners(gt_boxesR_theta)                
                gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))
                # gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))
                # gt_boxesRMem_end = nlu.get_ends_of_corner(gt_boxesRMem_corners)
            gt_boxesRMem_theta = utils_geom.transform_corners_to_boxes(gt_boxesRMem_corners)
            gt_boxesRUnp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z,Y,X))
            gt_boxesRUnp_end = nlu.get_ends_of_corner(gt_boxesRUnp_corners)
            
            if hyp.gt_rotate_combinations:
                gt_boxesX1_corners = __ub(utils_geom.apply_4x4(camX1_T_camRs, __pb(gt_boxesR_corners)))
                gt_boxesX1Unp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX1_corners),Z,Y,X))
                gt_boxesX1Unp_end = nlu.get_ends_of_corner(gt_boxesX1Unp_corners)
            
            gt_boxesX0_corners = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners)))
            gt_boxesX0Mem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX0_corners),Z2,Y2,X2))

            gt_boxesX0Mem_theta = utils_geom.transform_corners_to_boxes(gt_boxesX0Mem_corners)
            
            gt_boxesX0Mem_end = nlu.get_ends_of_corner(gt_boxesX0Mem_corners)
            gt_boxesX0_end = nlu.get_ends_of_corner(gt_boxesX0_corners)

            gt_cornersX0_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))

            rgb_camX0 = rgb_camXs[:,0]
            rgb_camX1 = rgb_camXs[:,1]
            # st()
            if hyp.dataset_name == "carla":
                summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamXtop', rgb_camtop, gt_boxescamXTop_corners, torch.from_numpy(scores), tids, pix_T_cams[:, 0])
            summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamX0', rgb_camX0, gt_boxesX0_corners, torch.from_numpy(scores), tids, pix_T_cams[:, 0])
            unps_vis = utils_improc.get_unps_vis(unpX0s_half, occX0s_half)
            unp_vis = torch.mean(unps_vis, dim=1)
            unps_visRs = utils_improc.get_unps_vis(unpRs_half, occRs_half)
            unp_visRs = torch.mean(unps_visRs, dim=1)
            unps_visRs_full = utils_improc.get_unps_vis(unpRs, occRs)
            unp_visRs_full = torch.mean(unps_visRs_full, dim=1)
            # print(gt_boxesRMem_end[0,0,:,1])
            # st()
            summ_writer.summ_box_mem_on_unp('eval_boxes/gt_boxesR_mem', unp_visRs , gt_boxesRMem_end, scores ,tids)
            
            unpX0s_half = torch.mean(unpX0s_half, dim=1)
            unpX0s_half = nlu.zero_out(unpX0s_half,gt_boxesX0Mem_end,scores)

            occX0s_half = torch.mean(occX0s_half, dim=1)
            occX0s_half = nlu.zero_out(occX0s_half,gt_boxesX0Mem_end,scores)            

            summ_writer.summ_unp('3D_inputs/unpX0s', unpX0s_half, occX0s_half)

        if hyp.do_feat:
            # st()
            # print(gt_boxesRMem_end[:,:1])
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D

            if hyp.imgnet:
                featXs_input = torch.cat([occXs, occXs*unpXs_pret], dim=2)                
            elif hyp.onlyocc:
                featXs_input = occXs
            else:
                featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)
            # it is useful to keep track of what was visible from each viewpoint
            freeXs_ = utils_vox.get_freespace(__p(xyz_camXs), __p(occXs_half))
            freeXs = __u(freeXs_)
            visXs = torch.clamp(occXs_half+freeXs, 0.0, 1.0)
            mask_ = None            
            if(type(mask_)!=type(None)):
                assert(list(mask_.shape)[2:5]==list(featXs_input_.shape)[2:5])
            featXs_, feat_loss = self.featnet(featXs_input_, summ_writer, mask=__p(occXs))#mask_)
            total_loss += feat_loss

            validXs = torch.ones_like(visXs)
            _validX00 = validXs[:,0:1]
            _validX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], validXs[:,1:])
            validX0s = torch.cat([_validX00, _validX01], dim=1)
            validRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, validXs)
            visRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, visXs)
            # _visX00 = visXs[:,0:1]
            # _visX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], visXs[:,1:])
            # visX0s = torch.cat([_visX00, _visX01], dim=1)
            
            featXs = __u(featXs_)
            _featX00 = featXs[:,0:1]
            _featX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], featXs[:,1:])
            featX0s = torch.cat([_featX00, _featX01], dim=1)

            emb3D_e = torch.mean(featX0s[:,1:], dim=1) # context
            vis3D_e_R = torch.max(visRs[:,1:], dim=1)[0]
            emb3D_g = featX0s[:,0] # obs
            vis3D_g_R = visRs[:,0] # obs #only select those indices which are visible and valid.
            validR_combo = torch.min(validRs,dim=1).values

            if hyp.do_eval_recall and summ_writer.save_this:
                results['emb3D_e'] = emb3D_e
                results['emb3D_g'] = emb3D_g

            if hyp.do_save_vis:
                # np.save('%s_rgb_%06d.npy' % (hyp.name, global_step), rgb_camRs[:,0].detach().cpu().numpy())
                imageio.imwrite('%s_rgb_%06d.png' % (hyp.name, global_step), np.transpose(utils_improc.back2color(rgb_camRs)[0,0].detach().cpu().numpy(), axes=[1, 2, 0]))
                np.save('%s_emb3D_g_%06d.npy' % (hyp.name, global_step), emb3D_e.detach().cpu().numpy())
            if not hyp.onlyocc:
                summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), valids=torch.unbind(validXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), valids=torch.unbind(torch.ones_like(validRs), dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validRs', torch.unbind(validRs, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e_R', vis3D_e_R, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g_R', vis3D_g_R, pca=False)
            if hyp.do_empty:
                empty_featXs_input = torch.cat([empty_occXs, empty_occXs*empty_unpXs], dim=2)
                empty_featXs_input_ = __p(empty_featXs_input)
                # it is useful to keep track of what was visible from each viewpoint
                empty_freeXs_ = utils_vox.get_freespace(__p(empty_xyz_camXs), __p(empty_occXs_half))
                empty_freeXs = __u(empty_freeXs_)

                empty_featXs_, empty_validXs_, empty_feat_loss = self.featnet(empty_featXs_input_, summ_writer, mask=__p(empty_occXs),prefix="empty_")

                empty_validXs = __u(empty_validXs_)
                empty__validX00 = empty_validXs[:,0:1]
                empty__validX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], empty_validXs[:,1:])
                empty_validX0s = torch.cat([empty__validX00, empty__validX01], dim=1)
                                
                empty_featXs = __u(empty_featXs_)
                empty__featX00 = empty_featXs[:,0:1]
                empty__featX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], empty_featXs[:,1:])
                empty_featX0s = torch.cat([empty__featX00, empty__featX01], dim=1)
            if hyp.profile_time:
                print("featnet time",time.time()-start_time)
        if hyp.store_obj:
            if hyp.store_ent_dis_obj:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_R = (emb3D_e_R + emb3D_g_R)/2
                # st()
                emb3D_R_object = nlu.crop_object_tensors(emb3D_R, gt_boxesRMem_end, scores)
                object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
                emb3D_R_object_ent = emb3D_R_object
                obj_shapes = [list(obj.shape[1:]) for obj in emb3D_R_object]
                emb3D_R_object = [torch.nn.functional.interpolate(obj_tensor.unsqueeze(0),size=[hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE],mode='nearest') for  obj_tensor in emb3D_R_object]
                # st()
                if len(emb3D_R_object) != 0:
                    content,style = self.munitnet.net.gen_a.encode(torch.cat(emb3D_R_object,dim=0))
                    object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
                    style_classes = []
                    content_classes = []
                    for object_class in object_classes:
                        style_classes.append(object_class.split("/")[1])
                        content_classes.append(object_class.split("/")[0])

                    results["aug_objects_content"] = content
                    results["aug_objects_style"] = style
                    results["content_classes"] = content_classes
                    results["style_classes"] = style_classes
                    results["obj_shapes"] = obj_shapes
                else:
                    results["aug_objects_content"] = []
                    results["aug_objects_style"] = []
                    results["content_classes"] = []
                    results["style_classes"] = []
                    results["obj_shapes"] = []    

                results["aug_objects"] = emb3D_R_object_ent
                results["classes"] = object_classes                

            if hyp.store_ent_obj:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_R = (emb3D_e_R + emb3D_g_R)/2
                # st()
                emb3D_R_object = nlu.crop_object_tensors(emb3D_R, gt_boxesRMem_end, scores)
                object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
                folder_name = "dump_obj/ent"
                results["aug_objects"] = emb3D_R_object
                results["classes"] = object_classes
            if hyp.store_dis_obj:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_R = (emb3D_e_R + emb3D_g_R)/2
                emb3D_R_object = nlu.crop_object_tensors(emb3D_R, gt_boxesRMem_end, scores)
                obj_shapes = [list(obj.shape[1:]) for obj in emb3D_R_object]
                emb3D_R_object = [torch.nn.functional.interpolate(obj_tensor.unsqueeze(0),size=[hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE],mode='nearest') for  obj_tensor in emb3D_R_object]
                # st()
                if len(emb3D_R_object) != 0:
                    content,style = self.munitnet.net.gen_a.encode(torch.cat(emb3D_R_object,dim=0))
                    object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
                    style_classes = []
                    content_classes = []
                    for object_class in object_classes:
                        style_classes.append(object_class.split("/")[1])
                        content_classes.append(object_class.split("/")[0])

                    results["aug_objects_content"] = content
                    results["aug_objects_style"] = style
                    results["content_classes"] = content_classes
                    results["style_classes"] = style_classes
                    results["obj_shapes"] = obj_shapes
                else:
                    results["aug_objects_content"] = []
                    results["aug_objects_style"] = []
                    results["content_classes"] = []
                    results["style_classes"] = []
                    results["obj_shapes"] = []                    
        else:
            if feed['set_name'] == "val" and hyp.do_match_det:
                hyp.B = 1
                # st()
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)                
                if hyp.remove_air:
                    occR_complete_to_create = occR_complete[:1]
                # st()
                emb3D_e_R = emb3D_e_R[:1]
                emb3D_g_R = emb3D_g_R[:1]
                if hyp.remove_air:
                    emb3D_e_R =  emb3D_e_R*occR_complete_to_create
                    emb3D_g_R = emb3D_g_R * occR_complete_to_create                    
                gt_boxesRMem_theta = gt_boxesRMem_theta[:1]
                scores = scores[:1]
                camR_T_camX0 = camR_T_camX0[:1]
                # st()
            # st()
            if (hyp.debug_match and hyp.aug_object_dis) or (hyp.do_match_det and hyp.aug_object_dis and feed['set_name'] == "train") or (hyp.do_match_det and hyp.aug_object_ent and feed['set_name'] == "train"):
                # st()
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)            
                folder_name = "dump_obj/ent"
                filenames = glob.glob(f"{folder_name}/*")
                emb3D_R = (emb3D_e_R + emb3D_g_R)/2
                # if global_step == 1:
                #     self.emb3D_R_object_tmp = nlu.crop_object_tensors(emb3D_R, gt_boxesRMem_end, scores)
                # st()
                all_objs = []           
                all_shape_vals = [] 
                all_aug_bboxes = []
                all_aug_score = []
                objects_content_taken = []
                objects_style_taken = []
                emb3D_R_to_create = emb3D_R[:1]
                gt_boxesRMem_theta_to_create = gt_boxesRMem_theta[:1]
                scores_to_create_2 = scores[:1]
                if hyp.remove_air:
                    occR_complete_to_create = occR_complete[:1]
                classes_to_create = classes[0]
                scores_to_create = scores[0]
                emb3D_R_empty = emb3D_R[1:]
                gt_boxesRMem_end_rounded = torch.round(gt_boxesRMem_end)
                scores_to_create_2 = scores[:1]
                boxes_to_create = gt_boxesRMem_end_rounded[:1]

                if hyp.aug_object_ent:
                    objects_being_replaced = nlu.crop_object_tensors(emb3D_R_to_create[:1], gt_boxesRMem_end[:1], scores[:1])
                    emb3D_R_aug_up = nlu.update_scene_with_object_crops(emb3D_R_empty, objects_being_replaced ,boxes_to_create, scores_to_create_2)
                else:
                    # for index_val,emb3D_R_val  in enumerate(emb3D_R):
                        # num_objects = random.randint(hyp.min_obj_aug,hyp.max_obj_aug)
                        # num_objects = 2
                    # aug_score = np.array([0]*hyp.max_obj_aug)
                    # aug_score[:num_objects] = 1
                        
                    # objects_content_indexes_taken = [random.choice(range(len(self.list_aug_content))) for i in range(num_objects)]
                    for index,class_val in enumerate(classes_to_create):
                        if scores_to_create[index] > 0:
                            content_class = class_val.split('/')[0]
                            style_class = class_val.split('/')[1]
                            content_val = self.dict_aug[content_class]
                            style_val = self.dict_aug[style_class]
                            objects_content_taken.append(content_val)
                            objects_style_taken.append(style_val)
                    objects_content_taken = torch.stack(objects_content_taken,dim=0)
                    objects_style_taken = torch.stack(objects_style_taken,dim=0)
                    objects_taken,_ = self.munitnet.net.gen_a.decode(objects_content_taken, objects_style_taken)
                    objects_being_replaced = nlu.crop_object_tensors(emb3D_R_to_create[:1], gt_boxesRMem_end[:1], scores[:1])
                    shape_vals = [obj.shape[1:] for obj in  objects_being_replaced]
                    # shape_vals = [torch.tensor([9, 9, 9]) for obj in  objects_being_replaced]
                    # st()
                    # if index_val == 0:
                    #     summ_writer.summ_text('aug_boxes/class_augmented', '//'.join(classes_taken))
                    # st()
                    # st()
                    objects_taken = [obj_tensor.unsqueeze(0) for index,obj_tensor in enumerate(objects_taken)]
                    objects_taken = [torch.nn.functional.interpolate(obj_tensor,size=list(shape_vals[index]),mode='nearest') for index,obj_tensor in enumerate(objects_taken)]
                    # objects_taken =
                    # st()
                    # scores_ex = np.where(scores[0]==1.)
                    # try:
                    #     boxes_ex = gt_boxesRMem_end[index_val][scores_ex]
                    # except Exception:
                    #     st()
                    # b_mask = nlu.create_binary_mask(boxes_ex,list(emb3D_R.shape[2:]))
                    y_max = gt_boxesRMem_end[0,0,1,1]
                    # num_objs = boxes_ex.shape[0]
                    
                    # all_shape_vals = all_shape_vals + shape_vals
                    # all_objs = all_objs + objects_taken
                    # aug_bboxes = nlu.sample_boxes(b_mask,num_objects,y_max=y_max,shape_val = shape_vals)

                    # aug_bboxes = np.pad(aug_bboxes,[[0,hyp.max_obj_aug-num_objects],[0,0],[0,0]])
                    # all_aug_score.append(aug_score)
                    # all_aug_bboxes.append(aug_bboxes)

                    # st()


                    # all_aug_bboxesX0corners = __ub_a(utils_geom.apply_4x4(camX0_T_camRs, __pb_a(all_aug_bboxesRcorners)))
                    # summ_writer.summ_box_by_corners('aug_boxes/aug_boxescamX0', rgb_camX0, all_aug_bboxesX0corners, torch.from_numpy(all_aug_score), tids, pix_T_cams[:, 0])
                    # st()
                    # try:
                    # st()
                    # if feed['set_name'] == "val":
                    #     st()
                    emb3D_R_aug_up = nlu.update_scene_with_object_crops(emb3D_R_empty, objects_taken ,boxes_to_create, scores_to_create_2)
                    # except Exception as e:
                    # st()
                emb3D_R_aug_diff = torch.abs(emb3D_R_aug_up - emb3D_R_to_create)
                summ_writer.summ_feat(f'aug_feat/og', emb3D_R_to_create)
                summ_writer.summ_feat(f'aug_feat/og_aug', emb3D_R_aug_up)
                summ_writer.summ_feat(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)
                # summ_writer.summ_diff_tensor(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)
                # st()
                emb3D_R_aug = torch.cat([emb3D_R_to_create,emb3D_R_aug_up],dim=0)
                if hyp.remove_air:
                    emb3D_R_to_create_ra =  emb3D_R_to_create*occR_complete_to_create
                    emb3D_R_aug_ra = emb3D_R_aug_up * occR_complete_to_create
                    emb3D_R_aug_diff_ra = torch.abs(emb3D_R_aug_ra - emb3D_R_to_create_ra)
                    summ_writer.summ_feat(f'aug_feat/og_ra', emb3D_R_to_create_ra)
                    summ_writer.summ_feat(f'aug_feat/og_aug_ra', emb3D_R_aug_ra)
                    summ_writer.summ_feat(f'aug_feat/og_aug_diff_ra', emb3D_R_aug_diff_ra)
                else:
                    emb3D_R_aug_ra = emb3D_R_aug_up
            
                if hyp.og_debug:
                    if hyp.remove_air:
                        emb3D_R_aug_ra = emb3D_R_to_create_ra
                    else:
                        emb3D_R_aug_ra = emb3D_R_to_create
                # st()
                # emb3D_R_aug = emb3D_R_aug.detach()


            elif hyp.aug_det and feed['set_name'] == "train":
                # st()
                # if hyp.aug_object_ent:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)            
                # folder_name = "dump_obj/ent"
                # filenames = glob.glob(f"{folder_name}/*")
                emb3D_R = (emb3D_e_R + emb3D_g_R)/2

                all_objs = []           
                all_shape_vals = [] 
                all_aug_bboxes = []
                all_aug_score = []
                for index_val,emb3D_R_val  in enumerate(emb3D_R):
                    num_objects = random.randint(hyp.min_obj_aug,hyp.max_obj_aug)
                    # num_objects = 1
                    if hyp.aug_object_ent_dis:
                        objects_indexes_taken = [random.choice(range(len(self.list_aug))) for i in range(num_objects)]
                        objects_taken = [self.list_aug[index] for index in objects_indexes_taken]
                        classes_taken = [self.list_aug_classes[index] for index in objects_indexes_taken]
                        shape_vals = [i.shape[1:] for i in objects_taken]                        
                        
                        tmp_shape_val = shape_vals
                        tmp_objects_taken =  objects_taken
                        tmp_classes_taken = classes_taken
                        #dis

                        #dis
                        num_objects_dis = random.randint(hyp.min_obj_aug_dis,hyp.max_obj_aug_dis)
                        if num_objects_dis> 0:
                            objects_content_indexes_taken = [random.choice(range(len(self.list_aug_content))) for i in range(num_objects_dis)]
                            objects_content_taken = [self.list_aug_content[index] for index in objects_content_indexes_taken]
                            classes_content_taken = [self.list_aug_classes_content[index] for index in objects_content_indexes_taken]

                            objects_style_indexes_taken = [random.choice(range(len(self.list_aug_style))) for i in range(num_objects_dis)]
                            objects_style_taken = [self.list_aug_style[index] for index in objects_style_indexes_taken]
                            classes_style_taken = [self.list_aug_classes_style[index] for index in objects_style_indexes_taken]

                            objects_content_taken = torch.stack(objects_content_taken,dim=0)
                            objects_style_taken = torch.stack(objects_style_taken,dim=0)

                            objects_taken,_ = self.munitnet.net.gen_a.decode(objects_content_taken, objects_style_taken)
                            classes_taken = ["/".join([classes_content_taken[index],classes_style_taken[index]]) for index in range(num_objects_dis)]
                            # class_content = classes_content_taken[index]
                            # resize_rand = random.uniform(0.5,1.1)
                            shape_vals = [random.choice(self.list_aug_shapes[class_content])  for class_content in classes_content_taken]
                            shape_vals = [(np.array(shape_val)).astype(np.int).tolist() for shape_val in shape_vals]
                            # st()
                            if not hyp.rotate_aug and not hyp.shape_aug:
                                objects_taken = [obj_tensor.unsqueeze(0) for index,obj_tensor in enumerate(objects_taken)]
                                objects_taken = [torch.nn.functional.interpolate(obj_tensor,size=list(shape_vals[index]),mode='nearest') for index,obj_tensor in enumerate(objects_taken)]
                                objects_taken = [obj_tensor.squeeze(0) for index,obj_tensor in enumerate(objects_taken)]

                            shape_vals = tmp_shape_val + shape_vals
                            objects_taken = tmp_objects_taken + objects_taken
                            classes_taken = tmp_classes_taken + classes_taken
                            num_objects = num_objects + num_objects_dis
                        else:
                            shape_vals = tmp_shape_val
                            objects_taken = tmp_objects_taken
                            classes_taken = tmp_classes_taken
                            num_objects = num_objects

                    elif hyp.aug_object_ent:
                        # st()
                        objects_indexes_taken = [random.choice(range(len(self.list_aug))) for i in range(num_objects)]
                        objects_taken = [self.list_aug[index] for index in objects_indexes_taken]
                        classes_taken = [self.list_aug_classes[index] for index in objects_indexes_taken]
                        shape_vals = [i.shape[1:] for i in objects_taken]
                    elif hyp.aug_object_dis:
                        objects_content_indexes_taken = [random.choice(range(len(self.list_aug_content))) for i in range(num_objects)]
                        objects_content_taken = [self.list_aug_content[index] for index in objects_content_indexes_taken]
                        classes_content_taken = [self.list_aug_classes_content[index] for index in objects_content_indexes_taken]

                        objects_style_indexes_taken = [random.choice(range(len(self.list_aug_style))) for i in range(num_objects)]
                        objects_style_taken = [self.list_aug_style[index] for index in objects_style_indexes_taken]
                        classes_style_taken = [self.list_aug_classes_style[index] for index in objects_style_indexes_taken]

                        objects_content_taken = torch.stack(objects_content_taken,dim=0)
                        objects_style_taken = torch.stack(objects_style_taken,dim=0)

                        objects_taken,_ = self.munitnet.net.gen_a.decode(objects_content_taken, objects_style_taken)
                        classes_taken = ["/".join([classes_content_taken[index],classes_style_taken[index]]) for index in range(num_objects)]
                        # class_content = classes_content_taken[index]
                        resize_rand = random.uniform(0.5,1.1)
                        shape_vals = [random.choice(self.list_aug_shapes[class_content])  for class_content in classes_content_taken]
                        shape_vals = [(np.array(shape_val)*resize_rand).astype(np.int).tolist() for shape_val in shape_vals]
                        # st()
                    if hyp.aug_object_ent_dis:
                        aug_score = np.array([0]*(hyp.max_obj_aug+hyp.max_obj_aug_dis))
                    else:
                        aug_score = np.array([0]*hyp.max_obj_aug)
                    aug_score[:num_objects] = 1
                    if index_val == 0:
                        summ_writer.summ_text('aug_boxes/class_augmented', '//'.join(classes_taken))
                    # st()
                    # st()
                    if hyp.rotate_aug:
                        objects_taken_filtered = []
                        for index,obj_tensor in enumerate(objects_taken):
                            obj_tensor_shape = shape_vals[index]
                            obj_tensor = torch.nn.functional.interpolate(obj_tensor.unsqueeze(0),size=[hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE],mode='nearest')
                            rot_ang = torch.tensor([random.randint(0,35)])
                            obj_tensor_rotated = self.mbr.rotateTensorToPose(obj_tensor,rot_ang)
                            if not hyp.shape_aug:
                                obj_tensor_rotated = torch.nn.functional.interpolate(obj_tensor_rotated,size=list(obj_tensor_shape),mode='nearest')
                            objects_taken_filtered.append(obj_tensor_rotated)
                        objects_taken = objects_taken_filtered
                    else:
                        objects_taken = [obj_tensor.unsqueeze(0) for index,obj_tensor in enumerate(objects_taken)]
                        if  hyp.aug_object_dis and not hyp.shape_aug:
                            objects_taken = [torch.nn.functional.interpolate(obj_tensor,size=list(shape_vals[index]),mode='nearest') for index,obj_tensor in enumerate(objects_taken)]
                            # objects_taken = 
                    # st()
                    # check this shouldn't be 0
                    scores_ex = np.where(scores[0]==1.)
                    try:
                        boxes_ex = gt_boxesRMem_end[index_val][scores_ex]
                    except Exception:
                        st()
                    b_mask = nlu.create_binary_mask(boxes_ex,list(emb3D_R.shape[2:]))


                    if hyp.shape_aug:
                        # resize_rand = random.uniform(0.5,1.1)
                        resize_rand = [random.uniform(0.7,1.2)  for shape_val in shape_vals]
                        shape_vals = [(np.array(shape_val)*resize_rand[ind]).astype(np.int).tolist() for ind,shape_val in enumerate(shape_vals)]
                        objects_taken = [torch.nn.functional.interpolate(obj_tensor,size=shape_vals[index],mode='nearest') for index,obj_tensor in enumerate(objects_taken)]

                    if hyp.aug_bbox_ymax != None: # Hardcode for empty scenes.
                        y_max = torch.tensor(hyp.aug_bbox_ymax).cuda() #35.5500
                    else:
                        y_max = boxes_ex[0,1,1]

                    all_shape_vals = all_shape_vals + shape_vals
                    all_objs = all_objs + objects_taken
                    aug_bboxes = nlu.sample_boxes(b_mask,num_objects,y_max=y_max,shape_val = shape_vals)
                    if hyp.aug_object_ent_dis:
                        aug_bboxes = np.pad(aug_bboxes,[[0,(hyp.max_obj_aug + hyp.max_obj_aug_dis)-num_objects],[0,0],[0,0]])                        
                    else:
                        aug_bboxes = np.pad(aug_bboxes,[[0,hyp.max_obj_aug-num_objects],[0,0],[0,0]])
                    all_aug_score.append(aug_score)
                    all_aug_bboxes.append(aug_bboxes)
                # st()
                all_aug_bboxesRMemEnd = np.array(all_aug_bboxes)
                all_aug_score = np.array(all_aug_score)
                all_aug_bboxesRMemEnd = torch.from_numpy(all_aug_bboxesRMemEnd).cuda().to(torch.float) 
                all_aug_score = np.array(all_aug_score)
                all_aug_bboxesREnd = __ub_a(utils_vox.Mem2Ref(__pb_a(all_aug_bboxesRMemEnd),Z2,Y2,X2))
                all_aug_bboxesRtheta = nlu.get_alignedboxes2thetaformat(all_aug_bboxesREnd) #torch.Size([2, 3, 9])
                all_aug_bboxesRcorners = utils_geom.transform_boxes_to_corners(all_aug_bboxesRtheta)
                all_aug_bboxesX0corners = __ub_a(utils_geom.apply_4x4(camX0_T_camRs, __pb_a(all_aug_bboxesRcorners)))

                summ_writer.summ_box_by_corners('aug_boxes/aug_boxescamX0', rgb_camX0, all_aug_bboxesX0corners, torch.from_numpy(all_aug_score), tids, pix_T_cams[:, 0])
                if hyp.debug_add:
                    emb3D_R_aug_old = emb3D_R
                # st()
                if hyp.add_random_noise:
                    all_objs = [obj + torch.normal(0.00000, 0.00005, size=obj.shape).cuda() for obj in all_objs]
                    # st()
                try:
                    emb3D_R_aug = nlu.update_scene_with_object_crops(emb3D_R, all_objs ,all_aug_bboxesRMemEnd, all_aug_score)
                except Exception as e:
                    st()
                # st()
                if hyp.remove_air:
                    occ_R_pred_ = self.occnet.predict(emb3D_R_aug,"aug_occ",summ_writer)
                    emb3D_R_aug = emb3D_R_aug *occ_R_pred_
                    

                if hyp.do_smoothnet or hyp.smoothness_with_noloss:
                    emb3D_R_aug, smoothness_loss = self.smoothnet(emb3D_R_aug, all_aug_bboxesRMemEnd, all_aug_score)
                    if not hyp.smoothness_with_noloss:
                        summ_writer.summ_scalar('smoothness_loss', smoothness_loss)
                        total_loss += smoothness_loss

                emb3D_R_aug = emb3D_R_aug.detach()
                emb3D_R_aug_old = emb3D_R
                if hyp.debug_add or True:
                    # st()
                    emb3D_R_aug_diff = torch.abs(emb3D_R_aug - emb3D_R_aug_old)
                    summ_writer.summ_feat(f'aug_feat/og', emb3D_R_aug_old)
                    summ_writer.summ_feat(f'aug_feat/og_aug', emb3D_R_aug)
                    summ_writer.summ_feat(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)


                all_gt_boxesR_end_aug = []
                all_gt_scores = []
                # st()
                for index_b in range(B):
                    gt_boxesR_end_filtered = gt_boxesR_end[index_b][np.where(scores[index_b]==1.)]
                    all_aug_bboxesREnd_filtered = all_aug_bboxesREnd[index_b][np.where(all_aug_score[index_b]==1.) ]
                    gt_boxesR_end_aug = torch.cat([gt_boxesR_end_filtered,all_aug_bboxesREnd_filtered],dim=0)
                    N_val = gt_boxesR_end_aug.shape[0]
                    gt_scores = np.zeros([hyp.N])
                    gt_scores[:N_val] = 1.0
                    try:
                        gt_boxesR_end_aug =  torch.cat([gt_boxesR_end_aug,torch.zeros([hyp.N-N_val,2,3]).cuda()],dim=0)
                    except: 
                        st()
                    all_gt_boxesR_end_aug.append(gt_boxesR_end_aug)
                    all_gt_scores.append(gt_scores)

                all_gt_boxesR_end_aug = torch.stack(all_gt_boxesR_end_aug)
                all_gt_scores = np.stack(all_gt_scores)
                all_gt_boxesR_theta_aug = nlu.get_alignedboxes2thetaformat(all_gt_boxesR_end_aug) #torch.Size([2, 3, 9])
                all_gt_boxesR_corners_aug = utils_geom.transform_boxes_to_corners(all_gt_boxesR_theta_aug)
                all_gt_boxesX0_corners_aug = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(all_gt_boxesR_corners_aug)))
                all_gt_boxesRMem_corners_aug = __ub(utils_vox.Ref2Mem(__pb(all_gt_boxesR_corners_aug),Z2,Y2,X2))
                all_gt_boxesRMem_theta_aug = utils_geom.transform_corners_to_boxes(all_gt_boxesRMem_corners_aug)
                # st()
                summ_writer.summ_box_by_corners('aug_boxes/all_aug_boxescamX0', rgb_camX0, all_gt_boxesX0_corners_aug, torch.from_numpy(all_gt_scores), tids, pix_T_cams[:, 0])


        if hyp.do_munit:

            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            if hyp.do_munit_det or hyp.do_munit_fewshot:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_R = emb3D_e_R
                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
                emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2
                # st()                
                content,style = self.munitnet.net.gen_a.encode(emb3D_R_object)
                objects_taken,_ = self.munitnet.net.gen_a.decode(content, style)
                styles = style
                contents = content
                # st()
            elif hyp.do_3d_style_munit:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_R = emb3D_e_R
                # st()
                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
                emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2

                camX1_T_R = camXs_T_camRs[:,1]
                camX0_T_R = camXs_T_camRs[:,0]            
                assert hyp.B == 2
                assert emb3D_e_R_object.shape[0] == 2
                munit_loss, sudo_input_0, sudo_input_1, recon_input_0, recon_input_1, sudo_input_0_cycle, sudo_input_1_cycle, styles , contents, adin = self.munitnet(emb3D_R_object[0:1], emb3D_R_object[1:2])
                if hyp.replace_sc:
                    adin_cs_a, adin_cs_b = adin
                    c_a, s_a_prime = adin_cs_a
                    c_b, s_b_prime = adin_cs_b
                    contents = [c_a,c_b]
                    styles = [s_a_prime, s_b_prime]
                # st()
                if hyp.store_content_style_range:
                    if self.max_content == None:
                        self.max_content = torch.zeros_like(contents[0][0]).cuda() - 100000000
                    if self.min_content == None:
                        self.min_content = torch.zeros_like(contents[0][0]).cuda() + 100000000
                    if self.max_style == None:
                        self.max_style = torch.zeros_like(styles[0][0]).cuda() - 100000000
                    if self.min_style == None:
                        self.min_style = torch.zeros_like(styles[0][0]).cuda() + 100000000
                    self.max_content = torch.max(torch.max(self.max_content, contents[0][0]), contents[1][0])
                    self.min_content = torch.min(torch.min(self.min_content, contents[0][0]), contents[1][0])
                    self.max_style = torch.max(torch.max(self.max_style, styles[0][0]), styles[1][0])
                    self.min_style = torch.min(torch.min(self.min_style, styles[0][0]), styles[1][0])

                    data_to_save = {'max_content': self.max_content.cpu().numpy(),'min_content': self.min_content.cpu().numpy(),
                    'max_style': self.max_style.cpu().numpy(), 'min_style': self.min_style.cpu().numpy()}
                    with open('content_style_range.p', 'wb') as f:
                        pickle.dump(data_to_save, f)
                elif hyp.is_contrastive_examples:
                    #Normalize
                    if hyp.normalize_contrast:
                        content0 = (contents[0]-self.min_content)/(self.max_content-self.min_content + 1e-5)
                        content1 = (contents[1]-self.min_content)/(self.max_content-self.min_content + 1e-5)
                        style0 = (styles[0]-self.min_style)/(self.max_style-self.min_style + 1e-5)
                        style1 = (styles[1]-self.min_style)/(self.max_style-self.min_style + 1e-5)
                    else:
                        content0 = contents[0]
                        content1 = contents[1]
                        style0 = styles[0]
                        style1 = styles[1]

                    # euclid_dist_content = torch.sum(torch.sqrt((content0 - content1)**2))/torch.prod(torch.tensor(content0.shape))
                    # euclid_dist_style = torch.sum(torch.sqrt((style0-style1)**2))/torch.prod(torch.tensor(style0.shape))
                    euclid_dist_content = (content0 - content1).norm(2) / (content0.numel())
                    euclid_dist_style = (style0 - style1).norm(2) / (style0.numel())

                    content_0_pooled = torch.mean(content0.reshape(list(content0.shape[:2]) + [-1]), dim=-1)
                    content_1_pooled = torch.mean(content1.reshape(list(content1.shape[:2]) + [-1]), dim=-1)

                    euclid_dist_content_pooled = (content_0_pooled - content_1_pooled).norm(2) / (content_0_pooled.numel())


                    content_0_normalized = content0/content0.norm()
                    content_1_normalized = content1/content1.norm()

                    style_0_normalized = style0/style0.norm()
                    style_1_normalized = style1/style1.norm()

                    content_0_pooled_normalized = content_0_pooled/content_0_pooled.norm()
                    content_1_pooled_normalized = content_1_pooled/content_1_pooled.norm()

                    cosine_dist_content = torch.sum(content_0_normalized*content_1_normalized)
                    cosine_dist_style = torch.sum(style_0_normalized*style_1_normalized)
                    cosine_dist_content_pooled = torch.sum(content_0_pooled_normalized*content_1_pooled_normalized)

                    print("euclid dist [content, pooled-content, style]: ", euclid_dist_content, euclid_dist_content_pooled, euclid_dist_style)
                    print("cosine sim [content, pooled-content, style]: ", cosine_dist_content, cosine_dist_content_pooled, cosine_dist_style)
                # st()
            if hyp.run_few_shot_on_munit: 
                # st()
                if (global_step % 300) == 1 or (global_step % 300) == 0:
                    # print("Emptying \n")
                    wrong = False
                    try:
                        precision_style = float(self.tp_style) /self.all_style
                        precision_content = float(self.tp_content) /self.all_content
                    except ZeroDivisionError:
                        wrong = True
                    # st()
                    if not wrong:
                        summ_writer.summ_scalar('precision/unsupervised_precision_style', precision_style)
                        summ_writer.summ_scalar('precision/unsupervised_precision_content', precision_content)
                        # st()
                    self.embed_list_style = defaultdict(lambda:[])
                    self.embed_list_content = defaultdict(lambda:[])
                    self.tp_style = 0
                    self.all_style = 0
                    self.tp_content = 0
                    self.all_content = 0
                    self.check = False
                elif not self.check and not nlu.check_fill_dict(self.embed_list_content,self.embed_list_style):
                    print("Filling \n")
                    for index,class_val in enumerate(object_classes):
                        # class_val = class_val[0]
                        if hyp.dataset_name == "clevr_vqa":
                            class_val_content, class_val_style = class_val.split("/")
                        else:
                            class_val_content, class_val_style = [class_val.split("/")[0],class_val.split("/")[0]]
                            # if class_val_content == "green_apple" or class_val_content == "appricot" or class_val_content == 'red_apple' or class_val_content == 'pear':
                            #     class_val_content = 'apple'
                            # if class_val_content == "small_banana" or class_val_content == "big_banana" or class_val_content == 'red_apple':
                            #     class_val_content = 'banana'
                            # if class_val_content == "tomato" or class_val_content == "avocado" or class_val_content == 'yellow_lemon' or class_val_content == 'green_lemon' or class_val_content ==  'red_peach':
                            #     class_val_content = 'lemon'                                
                            # if class_val_content == "green_grapes" or class_val_content == "red_grapes" or class_val_content == 'black_grapes':
                            #     class_val_content = 'grape'
                            # st()
                        # st()
                        print(len(self.embed_list_style.keys()),"style class",len(self.embed_list_content),"content class",self.embed_list_content.keys())
                        if len(self.embed_list_style[class_val_style]) < hyp.few_shot_nums:
                            self.embed_list_style[class_val_style].append(styles[index].squeeze())
                        if len(self.embed_list_content[class_val_content]) < hyp.few_shot_nums:
                            if hyp.avg_3d:
                                content_val = contents[index]
                                content_val = torch.mean(content_val.reshape([content_val.shape[1],-1]),dim=-1)
                                # st()
                                self.embed_list_content[class_val_content].append(content_val)
                            else:
                                self.embed_list_content[class_val_content].append(contents[index].reshape([-1]))
                else:
                    self.check = True
                    # print("Evaluating \n")
                    # print(str(len(list(self.embed_list_style.keys())))+"\n")
                    # print()
                    try:
                        print(float(self.tp_content) /self.all_content)
                        print(float(self.tp_style) /self.all_style)
                    except Exception as e:
                        pass
                    average = True
                    if average:
                        for key,val in self.embed_list_style.items():
                            if isinstance(val,type([])):
                                self.embed_list_style[key] = torch.mean(torch.stack(val,dim=0),dim=0)

                        for key,val in self.embed_list_content.items():
                            if isinstance(val,type([])):
                                self.embed_list_content[key] = torch.mean(torch.stack(val,dim=0),dim=0)
                    else:
                        for key,val in self.embed_list_style.items():
                            if isinstance(val,type([])):
                                self.embed_list_style[key] = torch.stack(val,dim=0)

                        for key,val in self.embed_list_content.items():
                            if isinstance(val,type([])):
                                self.embed_list_content[key] = torch.stack(val,dim=0)
                    for index,class_val in enumerate(object_classes):
                        class_val = class_val
                        if hyp.dataset_name == "clevr_vqa":
                            class_val_content, class_val_style = class_val.split("/")
                        else:
                            class_val_content, class_val_style = [class_val.split("/")[0],class_val.split("/")[0]]
                            # if class_val_content == "green_apple" or class_val_content == "appricot" or class_val_content == 'red_apple' or class_val_content == 'pear':
                            #     class_val_content = 'apple'
                            # if class_val_content == "small_banana" or class_val_content == "big_banana" or class_val_content == 'red_apple':
                            #     class_val_content = 'banana'
                            # if class_val_content == "tomato" or class_val_content == "avocado" or class_val_content == 'yellow_lemon' or class_val_content == 'green_lemon' or class_val_content ==  'red_peach':
                            #     class_val_content = 'lemon'                                
                            # if class_val_content == "green_grapes" or class_val_content == "red_grapes" or class_val_content == 'black_grapes':
                            #     class_val_content = 'grape'
                        # style
                        # st()
                        style_val = styles[index].squeeze().unsqueeze(0)
                        if not average:
                            embed_list_val_style = torch.cat(list(self.embed_list_style.values()),dim=0)
                            embed_list_key_style = list(np.repeat(np.expand_dims(list(self.embed_list_style.keys()),1),hyp.few_shot_nums,1).reshape([-1]))
                        else:
                            embed_list_val_style = torch.stack(list(self.embed_list_style.values()),dim=0)
                            embed_list_key_style = list(self.embed_list_style.keys())
                        embed_list_val_style = utils_basic.l2_normalize(embed_list_val_style,dim=1).permute(1,0)
                        style_val = utils_basic.l2_normalize(style_val,dim=1)
                        scores_styles = torch.matmul(style_val,embed_list_val_style)
                        index_key = torch.argmax(scores_styles,dim=1).squeeze()
                        selected_class_style = embed_list_key_style[index_key]
                        self.styles_prediction[class_val_style].append(selected_class_style)
                        if class_val_style == selected_class_style:
                            self.tp_style += 1
                        self.all_style += 1
                        # content
                        # st()
                        if hyp.avg_3d:
                            content_val = contents[index]
                            content_val = torch.mean(content_val.reshape([content_val.shape[1],-1]),dim=-1).unsqueeze(0)
                        else:
                            content_val = contents[index].reshape([-1]).unsqueeze(0)
                        if not average:
                            embed_list_val_content = torch.cat(list(self.embed_list_content.values()),dim=0)
                            embed_list_key_content = list(np.repeat(np.expand_dims(list(self.embed_list_content.keys()),1),hyp.few_shot_nums,1).reshape([-1]))
                        else:
                            embed_list_val_content = torch.stack(list(self.embed_list_content.values()),dim=0)
                            embed_list_key_content = list(self.embed_list_content.keys())
                        embed_list_val_content = utils_basic.l2_normalize(embed_list_val_content,dim=1).permute(1,0)
                        content_val = utils_basic.l2_normalize(content_val,dim=1)
                        scores_content = torch.matmul(content_val,embed_list_val_content)
                        index_key = torch.argmax(scores_content,dim=1).squeeze()
                        selected_class_content = embed_list_key_content[index_key]
                        self.content_prediction[class_val_content].append(selected_class_content)
                        if class_val_content == selected_class_content:
                            self.tp_content += 1

                        self.all_content += 1
            if hyp.do_munit_det or hyp.do_munit_fewshot:
                # st()
                recon_emb3D_R = nlu.update_scene_with_objects(emb3D_R, objects_taken, gt_boxesRMem_end, scores)
                emb3D_R_aug_diff = torch.abs(emb3D_R - recon_emb3D_R)
                summ_writer.summ_feat(f'aug_feat/og', emb3D_R)
                summ_writer.summ_feat(f'aug_feat/og_gen', recon_emb3D_R)
                summ_writer.summ_feat(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)                    
                if hyp.remove_air:
                    emb3D_R = emb3D_R*occR_complete
                    recon_emb3D_R = recon_emb3D_R*occR_complete
                    emb3D_R_aug_diff = torch.abs(emb3D_R - recon_emb3D_R)
                    summ_writer.summ_feat(f'aug_feat/og_ra', emb3D_R)
                    summ_writer.summ_feat(f'aug_feat/og_gen_ra', recon_emb3D_R)
                    summ_writer.summ_feat(f'aug_feat/og_aug_diff_ra', emb3D_R_aug_diff)                                        
            else:
                munit_loss = hyp.munit_loss_weight*munit_loss

                recon_input_obj = torch.cat([recon_input_0, recon_input_1],dim=0)
                recon_emb3D_R = nlu.update_scene_with_objects(emb3D_R, recon_input_obj, gt_boxesRMem_end, scores)

                sudo_input_obj = torch.cat([sudo_input_0,sudo_input_1],dim=0)
                styled_emb3D_R = nlu.update_scene_with_objects(emb3D_R, sudo_input_obj, gt_boxesRMem_end, scores)

                styled_emb3D_e_X1 = utils_vox.apply_4x4_to_vox(camX1_T_R, styled_emb3D_R)
                styled_emb3D_e_X0 = utils_vox.apply_4x4_to_vox(camX0_T_R, styled_emb3D_R)

                emb3D_e_X1 = utils_vox.apply_4x4_to_vox(camX1_T_R, recon_emb3D_R)
                emb3D_e_X0 = utils_vox.apply_4x4_to_vox(camX0_T_R, recon_emb3D_R)


                emb3D_e_X1_og = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R)
                emb3D_e_X0_og = utils_vox.apply_4x4_to_vox(camX0_T_R, emb3D_R)



                emb3D_R_aug_diff = torch.abs(emb3D_R - recon_emb3D_R)
                if hyp.remove_air:
                    recon_emb3D_R = recon_emb3D_R*occR_complete
                    emb3D_R = emb3D_R*occR_complete
                summ_writer.summ_feat(f'aug_feat/og', emb3D_R)
                summ_writer.summ_feat(f'aug_feat/og_gen', recon_emb3D_R)
                summ_writer.summ_feat(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)
                # summ_writer.summ_diff_tensor(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)                    
                # st()

                if hyp.cycle_style_view_loss:
                    sudo_input_obj_cycle = torch.cat([sudo_input_0_cycle,sudo_input_1_cycle],dim=0)
                    styled_emb3D_R_cycle = nlu.update_scene_with_objects(emb3D_R, sudo_input_obj_cycle, gt_boxesRMem_end, scores)

                    styled_emb3D_e_X0_cycle = utils_vox.apply_4x4_to_vox(camX0_T_R, styled_emb3D_R_cycle)
                    styled_emb3D_e_X1_cycle = utils_vox.apply_4x4_to_vox(camX1_T_R, styled_emb3D_R_cycle)
                summ_writer.summ_scalar('munit_loss', munit_loss.cpu().item())
                total_loss += munit_loss

            if hyp.do_2d_style_munit:
                # Assumptions: Batch size is 2 and there is only 1 object in scene.
                assert hyp.B == 2
                pix_T_cam = pix_T_cams[:, 0]
                gt_boxesX0_corners = gt_boxesX0_corners[:,:1]
                B2, N, D, E = list(gt_boxesX0_corners.shape)
                corners_cam_ = torch.reshape(gt_boxesX0_corners, [B, N*8, 3])
                corners_pix_ = utils_geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
                corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
                corners_pix = corners_pix.squeeze(1)
                ys = corners_pix[:,:,0]
                xs = corners_pix[:,:,1]
                ymax, ymin, xmax, xmin = torch.max(ys, dim=1)[0].int(), torch.min(ys, dim=1)[0].int(), torch.max(xs, dim=1)[0].int(), torch.min(xs, dim=1)[0].int()
                # ymax, ymin, xmax, xmin = int(ymax),int(ymin),int(xmax),int(xmin)
                obj_b0 = rgb_camX0[0,:,xmin[0]:xmax[0]+1, ymin[0]:ymax[0]+1]
                obj_b0 = obj_b0.unsqueeze(0)
                obj_b0 = F.interpolate(obj_b0, size=(32,32), mode="bilinear")

                obj_b1 = rgb_camX0[1,:,xmin[1]:xmax[1]+1, ymin[1]:ymax[1]+1]
                obj_b1 = obj_b1.unsqueeze(0)
                obj_b1 = F.interpolate(obj_b1, size=(32,32), mode="bilinear")

                munit_loss, sudo_input_0, sudo_input_1, recon_input_0, recon_input_1 = self.munitnet(obj_b0, obj_b1)
                gt_rgx_x00 = torch.cat([obj_b0[0], obj_b1[0]], dim=-1)
                recon_rgb = torch.cat([recon_input_0[0], recon_input_1[0]], dim=-1)
                sudo_styletransfer_rgb = torch.cat([sudo_input_0[0], sudo_input_1[0]], dim=-1)
                complete_vis = torch.cat([gt_rgx_x00, recon_rgb, sudo_styletransfer_rgb], dim=1)
                summ_writer.summ_rgb('munit/munit_recons_vis', complete_vis.unsqueeze(0))
                summ_writer.summ_scalar('munit_loss', munit_loss.cpu().item())
                total_loss += munit_loss


        # st()
        # if feed['set_name'] == "test":
        #     st()
        if (hyp.do_pixor_det or hyp.do_det) and not hyp.store_obj:
                if hyp.do_munit:
                    if feed['set_name'] == "train":
                        featR = recon_emb3D_R
                    else:
                        featR = (emb3D_e_R +emb3D_g_R)/2
                elif hyp.do_match_det:
                    if feed['set_name'] == "train":
                        featR = emb3D_R_aug_ra
                        hyp.B = 1
                    elif feed['set_name'] == "val":
                        featR = (emb3D_e_R + emb3D_g_R)/2
                elif (hyp.aug_det and feed['set_name'] == "train"):
                    featR = emb3D_R_aug
                elif (hyp.aug_det and feed['set_name'] == "val"):
                    emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                    emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)            
                    featR = (emb3D_e_R + emb3D_g_R)/2                         
                    if hyp.remove_air:
                        occ_R_pred_ = self.occnet.predict(featR,"aug_occ",summ_writer)
                        featR = featR *occ_R_pred_                    
                else:
                    emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                    emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)            
                    if hyp.single_view:
                        featR = emb3D_e_R
                    else:
                        featR = (emb3D_e_R + emb3D_g_R)/2            
                # featRs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, featXs)
                # featR = torch.mean(featRs, dim=1)
                self.unp_visRs = unp_visRs
                self.camR_T_camX0 = camR_T_camX0
                self.N_det = hyp.K*2
                if hyp.do_match_det and feed['set_name'] == "train":
                    self.axboxlist_memR = utils_geom.inflate_to_axis_aligned_boxlist(gt_boxesRMem_theta_to_create)
                    self.scorelist_s = torch.from_numpy(scores_to_create_2).cuda().to(torch.float)
                    # st()
                elif (hyp.aug_det and feed['set_name'] == "train"):
                    self.axboxlist_memR = utils_geom.inflate_to_axis_aligned_boxlist(all_gt_boxesRMem_theta_aug)
                    self.scorelist_s = torch.from_numpy(all_gt_scores).cuda().to(torch.float)
                else:
                    self.emb3D_e = emb3D_e
                    self.emb3D_g = emb3D_g
                    self.axboxlist_memR = utils_geom.inflate_to_axis_aligned_boxlist(gt_boxesRMem_theta)
                    self.scorelist_s = torch.from_numpy(scores).cuda().to(torch.float)
                if hyp.do_pixor_det:
                    self.pixor_targets = pixor.get_pixor_regression_targets(unp_visRs, summ_writer,self.geom['label_shape'],gt_boxesR_theta)

                    start_time = time.time()
                    self.pixor_boxes_theta_format,self.scorelist_g = self.decoder_boxes(self.pixor_targets, self.geom['label_shape'])
                    # print("first decoder",time.time()-start_time)

                    self.pixor_boxesMem_theta_format = utils_vox.convert_boxlist_camR_to_memR(self.pixor_boxes_theta_format, hyp.Z2, hyp.Y2, hyp.X2)
                    summ_writer.summ_box_mem_on_mem('pixor/gt_boxesR_mem_regen', unp_visRs, self.pixor_boxesMem_theta_format ,torch.from_numpy(scores),torch.ones([hyp.B,hyp.N],dtype=torch.int32))
                    predictions = self.pixor(featR)
                    start_time = time.time()
                    # st()
                    self.predicted_boxes_theta_format,scorelist_e = self.decoder_boxes(predictions, self.geom['label_shape'])
                    # print("second decoder",time.time()-start_time)
                    self.predicted_boxesMem_theta_format = utils_vox.convert_boxlist_camR_to_memR(self.predicted_boxes_theta_format, hyp.Z2, hyp.Y2, hyp.X2)
                    summ_writer.summ_box_mem_on_mem('pixor/pred_boxesR_mem', unp_visRs, self.predicted_boxesMem_theta_format,scorelist_e,torch.ones([hyp.B,hyp.N],dtype=torch.int32))
                    detect_loss, cls, loc = self.loss_fn(predictions, self.pixor_targets)
                    scorelist_g = self.scorelist_g[0:1].detach().cpu().numpy()
                    boxlist_camR_e = self.predicted_boxes_theta_format
                    boxlist_camR_g = gt_boxesR_theta
                elif hyp.do_det and hyp.self_improve_iterate:
                    if hyp.exp_do:
                        gt_boxesMem_to_consider_after_q_distance_theta, gt_scoresMem_to_consider_after_q_distance, feat_mask,gt_boxesMem_to_consider_after_cs_theta,gt_scoresMem_to_consider_after_cs = self.evaluate_filter_boxes(gt_boxesRMem_theta,scores,featR,summ_writer)
                        if hyp.replace_with_cs:
                            gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_cs_theta
                            gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_cs

                        gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_q_distance_theta.detach()
                        gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_q_distance.detach().cuda()
                        axboxlist_memR_filtered = []
                        scorelist_s_filtered = []
                        feat_mask_filtered = []
                        filenames_e = []
                        filenames_g = []

                        zeroth_example_presence = False
                        for ind,score_index in enumerate(gt_scoresMem_to_consider_after_q_distance):
                            if (score_index == 1).any():
                                if ind == 0:
                                    zeroth_example_presence = True
                                axboxlist_memR_filtered.append(gt_boxesMem_to_consider_after_q_distance_theta[ind])
                                scorelist_s_filtered.append(gt_scoresMem_to_consider_after_q_distance[ind])
                                feat_mask_filtered.append(feat_mask[ind])
                                filenames_e.append(feed['filename_e'][ind])
                                filenames_g.append(feed['filename_g'][ind])

                        if len(axboxlist_memR_filtered) > 0:
                            axboxlist_memR_filtered = torch.stack(axboxlist_memR_filtered)
                            scorelist_s_filtered = torch.stack(scorelist_s_filtered)
                            feat_mask_filtered = torch.stack(feat_mask_filtered)

                            results["filtered_boxes"] = axboxlist_memR_filtered
                            results["gt_boxes"] = self.axboxlist_memR
                            results["gt_scores"] = self.scorelist_s
                            results["featR_masks"] = feat_mask_filtered
                            results["scores"] =scorelist_s_filtered
                            results["filenames_e"] = filenames_e
                            results["filenames_g"] = filenames_g
                        else:
                            results["filtered_boxes"] = None
                            results["featR_masks"] = None
                            results["filenames_e"] = None
                            results["filenames_g"] = None
                            results["gt_boxes"] = None
                            results["gt_scores"] = None
                        # boxlist_memR_e = gt_boxesMem_to_consider_after_q_distance_theta
                        # scorelist_e = gt_scoresMem_to_consider_after_q_distance

                        _, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                            gt_boxesMem_to_consider_after_q_distance_theta[:1],
                            gt_scoresMem_to_consider_after_q_distance[:1],
                            featR[:1],
                            summ_writer)
                        # st()
                        boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                        boxlist_camR_g = utils_vox.convert_boxlist_memR_to_camR(self.axboxlist_memR, hyp.Z2, hyp.Y2, hyp.X2)
                        summ_writer.summ_box_mem_on_mem('detnet/gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,hyp.N],dtype=torch.int32))
                        summ_writer.summ_box_mem_on_mem('detnet/pred_boxesR_mem', unp_visRs, boxlist_memR_e ,scorelist_e,torch.ones_like(scorelist_e,dtype=torch.int32))
                        scorelist_g = self.scorelist_s[0:1].detach().cpu().numpy()
                    
                    if hyp.max_do: 
                        self.axboxlist_memR = feed["sudo_gt_boxes"]
                        self.scorelist_s = feed["sudo_gt_scores"]
                        featR_mask = feed["feat_mask"]
                        summ_writer.summ_occ('detnet/mask_used', featR_mask)
                        if hyp.maskout:
                            detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                    self.axboxlist_memR,
                                    self.scorelist_s,
                                    featR,
                                    summ_writer,mask=featR_mask.squeeze(1))
                        else:
                            detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                self.axboxlist_memR,
                                self.scorelist_s,
                                featR,
                                summ_writer)
                        total_loss += detect_loss            
                        summ_writer.summ_box_mem_on_mem('detnet/sudo_gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,self.N_det],dtype=torch.int32))
                        self.axboxlist_memR = feed["gt_boxes"]
                        self.scorelist_s = feed["gt_scores"]
                        boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                        boxlist_camR_g = utils_vox.convert_boxlist_memR_to_camR(self.axboxlist_memR, hyp.Z2, hyp.Y2, hyp.X2)
                        summ_writer.summ_box_mem_on_mem('detnet/gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,self.N_det],dtype=torch.int32))
                        summ_writer.summ_box_mem_on_mem('detnet/pred_boxesR_mem', unp_visRs, boxlist_memR_e ,scorelist_e,torch.ones_like(scorelist_e,dtype=torch.int32))
                        scorelist_g = self.scorelist_s[0:1].detach().cpu().numpy()
                elif hyp.do_det:
                    if hyp.filter_boxes:
                        gt_boxesMem_to_consider_after_q_distance_theta,gt_scoresMem_to_consider_after_q_distance, feat_mask,gt_boxesMem_to_consider_after_cs_theta,gt_scoresMem_to_consider_after_cs = self.evaluate_filter_boxes(gt_boxesRMem_theta,scores,featR,summ_writer)
                        if hyp.replace_with_cs:
                            gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_cs_theta
                            gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_cs

                    if hyp.filter_boxes and hyp.self_improve_once:
                        gt_boxesMem_to_consider_after_q_distance_theta = gt_boxesMem_to_consider_after_q_distance_theta.detach()
                        gt_scoresMem_to_consider_after_q_distance = gt_scoresMem_to_consider_after_q_distance.detach().cuda()
                        featR_filtered = []
                        axboxlist_memR_filtered = []
                        scorelist_s_filtered = []
                        feat_mask_filtered = []
                        zeroth_example_presence = False
                        for ind,score_index in enumerate(gt_scoresMem_to_consider_after_q_distance):
                            if (score_index == 1).any():
                                if ind == 0:
                                    zeroth_example_presence = True
                                featR_filtered.append(featR[ind])
                                axboxlist_memR_filtered.append(gt_boxesMem_to_consider_after_q_distance_theta[ind])
                                scorelist_s_filtered.append(gt_scoresMem_to_consider_after_q_distance[ind])
                                feat_mask_filtered.append(feat_mask[ind])
                        if len(axboxlist_memR_filtered) > 0:
                            axboxlist_memR_filtered = torch.stack(axboxlist_memR_filtered)
                            scorelist_s_filtered = torch.stack(scorelist_s_filtered)
                            featR_filtered = torch.stack(featR_filtered)
                            feat_mask_filtered = torch.stack(feat_mask_filtered)
                            if hyp.maskout:
                                detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                        axboxlist_memR_filtered,
                                        scorelist_s_filtered,
                                        featR_filtered,
                                        summ_writer,mask=feat_mask_filtered.squeeze(1))
                            else:
                                detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                    axboxlist_memR_filtered,
                                    scorelist_s_filtered,
                                    featR_filtered,
                                    summ_writer)

                            total_loss += detect_loss                        
                        else:
                            hyp.sudo_backprop = False

                        if not zeroth_example_presence:
                            with torch.no_grad():
                                _, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                                    gt_boxesMem_to_consider_after_q_distance_theta[:1],
                                    gt_scoresMem_to_consider_after_q_distance[:1],
                                    featR[:1],
                                    summ_writer)
                    else:
                        detect_loss, boxlist_memR_e, scorelist_e, tidlist_e, sco, ove = self.detnet(
                            self.axboxlist_memR,
                            self.scorelist_s,
                            featR,
                            summ_writer)
                        if hyp.add_det_boxes:
                            for index in range(hyp.B):
                                tree_filename_curr = tree_seq_filename[index]
                                tree = trees[index]
                                tree.bbox_det = boxlist_memR_e[index]
                                tree.score_det = scorelist_e[index]
                                tree_filename_curr = join(hyp.root_dataset,tree_filename_curr)
                                pickle.dump(tree,open(tree_filename_curr,"wb"))
                            print("check")
                        total_loss += detect_loss
                    # st()
                    boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                    boxlist_camR_g = utils_vox.convert_boxlist_memR_to_camR(self.axboxlist_memR, hyp.Z2, hyp.Y2, hyp.X2)
                    summ_writer.summ_box_mem_on_mem('detnet/gt_boxesR_mem', unp_visRs, self.axboxlist_memR ,self.scorelist_s,torch.ones([hyp.B,hyp.N],dtype=torch.int32))
                    try:
                        summ_writer.summ_box_mem_on_mem('detnet/pred_boxesR_mem', unp_visRs, boxlist_memR_e ,scorelist_e,torch.ones_like(scorelist_e,dtype=torch.int32))
                    except Exception as e:
                        print('will handle this later')
                    scorelist_g = self.scorelist_s[0:1].detach().cpu().numpy()
                if hyp.only_cs_vis:
                    boxlist_memR_e = gt_boxesMem_to_consider_after_cs_theta
                    scorelist_e = gt_scoresMem_to_consider_after_cs
                    boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                if hyp.only_q_vis:
                    boxlist_memR_e = gt_boxesMem_to_consider_after_q_distance_theta
                    scorelist_e = gt_scoresMem_to_consider_after_q_distance
                    boxlist_camR_e = utils_vox.convert_boxlist_memR_to_camR(boxlist_memR_e, hyp.Z2, hyp.Y2, hyp.X2)
                lrtlist_camR_e = utils_geom.convert_boxlist_to_lrtlist(boxlist_camR_e)
                boxlist_e = boxlist_camR_e[0:1].detach().cpu().numpy()
                boxlist_g = boxlist_camR_g[0:1].detach().cpu().numpy()
                scorelist_e = scorelist_e[0:1].detach().cpu().numpy()
                # start_pixor_det = time.time()
                boxlist_e, boxlist_g, scorelist_e, scorelist_g = utils_eval.drop_invalid_boxes(
                    boxlist_e, boxlist_g, scorelist_e, scorelist_g)
                # st()
                # print(boxlist_e,"predboxes",boxlist_g,"gt_boxes")
                ious = [0.3, 0.4, 0.5, 0.6, 0.7]
                maps,precisions_avg,scores_pred_val,ious_found = utils_eval.get_mAP(boxlist_e, scorelist_e, boxlist_g, ious)
                results['maps'] = maps
                results['ious'] = ious
                # st()
                results['filenames'] = feed['filename_e']+feed['filename_g'] 
                results['summ'] = summ_writer
                # st()
                for ind, overlap in enumerate(ious):
                    summ_writer.summ_scalar('ap/%.2f_iou' % overlap, maps[ind])
                    summ_writer.summ_scalar('precision/%.2f_iou' % overlap, precisions_avg[ind])
                if hyp.self_improve_iterate:
                    if hyp.exp_do:
                        self.avg_ap.append(maps[2])
                        self.avg_precision.append(precisions_avg[2])
                        size = len(self.avg_ap)
                        if ((size+1) % 100) == 0.0:
                            summ_writer.summ_scalar('ap/AVG_0.5_iou' , np.mean(self.avg_ap))
                            summ_writer.summ_scalar('precision/AVG_0.5_iou' , np.mean(self.avg_precision))
                            self.avg_ap = []
                            self.avg_precision = []
                else:
                    self.avg_ap.append(maps[2])
                    self.avg_precision.append(precisions_avg[2])
                    # st()
                    if ((global_step+1) % 100) == 0.0:
                        summ_writer.summ_scalar('ap/AVG_0.5_iou' , np.mean(self.avg_ap))
                        summ_writer.summ_scalar('precision/AVG_0.5_iou' , np.mean(self.avg_precision))
                # st()


        if hyp.create_prototypes:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2
            self.create_protos.add_prototypes(emb3D_R_object,object_classes)

        if hyp.create_example_dict:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            # print(object_classes)

            if hyp.normalize_style:
                emb3D_e_R_object = utils_geom.in_un(emb3D_e_R_object)
                emb3D_g_R_object = utils_geom.in_un(emb3D_g_R_object)

            emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2

            minNum = 1
            if len(self.embed_dict.keys()) == self.minclasses:
                minNum = ((hyp.object_quantize_dictsize//self.minclasses)+1)
            if len(self.embed_list) == hyp.object_quantize_dictsize and len(self.embed_dict.keys()) == self.minclasses:
                embed_list = torch.stack(self.embed_list).cpu().numpy().reshape([hyp.object_quantize_dictsize,-1])
                filename_np = f'offline_obj_cluster/{hyp.feat_init}_cluster_centers_Example_{hyp.object_quantize_dictsize}.npy'
                print(filename_np)
                np.save(filename_np,embed_list)
                st()
                

            for index,class_val in enumerate(object_classes):
                if self.embed_dict[class_val] < minNum and len(self.embed_list) < hyp.object_quantize_dictsize:
                    self.embed_dict[class_val] += 1
                    self.embed_list.append(emb3D_R_object[index])
            # st()
            print("embed size",len(self.embed_list),"keys",self.embed_dict.keys(),"len keys",len(self.embed_dict.keys()))

        if hyp.learn_linear_embeddings:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)

            emb3D_R = emb3D_e_R
            try:
                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            except Exception as e:
                st()
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            results['emb3D_e'] = emb3D_e_R_object
            results['emb3D_g'] = emb3D_g_R_object
            loss_sup_emb, acc_sup_emb = self.supervised_protos(emb3D_e_R_object, object_classes)
            summ_writer.summ_scalar('supervised_embeddings_loss', loss_sup_emb.cpu().item())
            summ_writer.summ_scalar('supervised_embeddings_accuracy', acc_sup_emb.cpu().item())
            total_loss += hyp.supervised_embedding_loss_coeff*loss_sup_emb


        if hyp.object_quantize:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_R = emb3D_e_R
            try:
                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            except Exception as e:
                st()
            # st()
            if hyp.normalize_style:
                emb3D_e_R_object = utils_geom.in_un(emb3D_e_R_object)
                emb3D_g_R_object = utils_geom.in_un(emb3D_g_R_object)
            
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            emb3D_R_object = emb3D_e_R_object 
            results['emb3D_e'] = emb3D_e_R_object
            results['emb3D_g'] = emb3D_g_R_object

            if hyp.use_supervised:
                loss_quant, quantized, perplexity, encodings, skip = self.quantizer(emb3D_R_object,object_classes)
                encodings = self.quantizer_evaluator(emb3D_R_object,self.quantizer.embeddings)
            else:
                loss_quant, quantized, perplexity, encodings,object_classes, skip = self.quantizer(emb3D_R_object,object_classes)
            # tensor_to_dump = {'feat_input':featXs_input_,'feat_out':featXs_,'distance':self.quantizer.distane_val,'filename_e':feed['filename_e'],'filename_g':feed['filename_g']}
            # pickle.dump(tensor_to_dump,open("dump/tensors.p","wb"))
            # st()
            # if hyp.eval_quantize:
            #     _, _, _, encodings_e = self.quantizer(emb3D_e_R_object)
            #     e_indexes = torch.argmax(encodings_e,dim=1).cpu().numpy()
            #     _, _, _, encodings_g = self.quantizer(emb3D_g_R_object)
            #     g_indexes = torch.argmax(encodings_g,dim=1).cpu().numpy()
                
            #     for index in range(hyp.B):
            #         class_val = object_classes[index]
            #         e_i = e_indexes[index]
            #         g_i = g_indexes[index]
            #         self.info_dict[str(e_i)].append(class_val)
            #         self.info_dict[str(g_i)].append(class_val)

            #     if (global_step % 1000) == 0:
            #         scores_dict = {}
            #         most_freq_dict = {}
            #         scores_list = []
            #         for key,item in self.info_dict.items():
            #             most_freq_word = utils_basic.most_frequent(item)
            #             mismatch = 0 
            #             for i in item:
            #                 if i != most_freq_word:
            #                     mismatch += 1
            #             precision = float(len(item)- mismatch)/len(item)
            #             scores_dict[key] = precision
            #             most_freq_dict[key] = most_freq_word
            #             scores_list.append(precision)
            #         unsupervised_score = np.mean(scores_list)
            #         summ_writer.summ_scalar('precision/unsupervised_precision', unsupervised_score)
            #         self.info_dict = defaultdict(lambda:[])

            if not skip:
                e_indexes = torch.argmax(encodings,dim=1).cpu().numpy()
                for index in range(len(object_classes)):
                    class_val = object_classes[index]
                    e_i = e_indexes[index]
                    self.info_dict[str(e_i)].append(class_val)


                if hyp.from_sup or hyp.use_supervised:
                    total = 0
                    true_postive = 0
                    e_indexes_pred = e_indexes//(hyp.object_quantize_dictsize) 
                    for index,e_index_val in enumerate(e_indexes_pred):
                        total += 1
                        object_class_val = object_classes[index]
                        actual_val = hyp.labels.index(object_class_val)
                        if actual_val == e_index_val:
                            true_postive+=1
                    precision = float(true_postive)/float(total)
                    summ_writer.summ_scalar('precision/supervised_precision', precision)

                if (global_step % 1000) == 0:
                    scores_dict = {}
                    scores_dict_norm = {}
                    most_freq_dict = {}
                    scores_list = []
                    total_mismatch = 0 
                    total_obj = 0 
                    for key,item in self.info_dict.items():
                        most_freq_word = utils_basic.most_frequent(item)
                        mismatch = 0 
                        for i in item:
                            total_obj += 1
                            if i != most_freq_word:
                                mismatch += 1
                                total_mismatch += 1
                        precision = float(len(item)- mismatch)/len(item)
                        scores_dict[key] = precision
                        scores_dict_norm[key] = precision*len(item)
                        most_freq_dict[key] = most_freq_word
                        scores_list.append(precision)
                    final_precision = (total_obj - total_mismatch)/float(total_obj)
                    summ_writer.summ_scalar('precision/unsupervised_precision', final_precision)
                    self.info_dict = defaultdict(lambda:[])                

            if hyp.gt_rotate_combinations:
                quantized,best_rotated_inputs,quantized_unrotated,best_rotations_index = quantized
                clusters = torch.argmax(encodings,dim=1)
                if hyp.use_gt_centers:
                    unique_indexes = []
                    for oc_ind,oc in enumerate(object_classes):
                        if oc not in self.list_of_classes:
                            self.list_of_classes.append(oc)
                        oc_index  = self.list_of_classes.index(oc)
                        unique_indexes.append(oc_index)
                # st()
                if hyp.use_gt_centers:
                    info_text = ['C'+str(int(unique_indexes[ind]))+'_R'+str(int(i)*10) for ind,i in enumerate(best_rotations_index)] 
                else:
                    info_text = ['C'+str(int(clusters[ind]))+'_R'+str(int(i)*10) for ind,i in enumerate(best_rotations_index)] 
                # summ_writer.summ_box_by_corners_parses('scene_parse/boxescamX1', rgb_camX1, gt_boxesX1_corners,torch.from_numpy(scores), tids, pix_T_cams[:, 0],info_text)
                summ_writer.summ_box_by_corners_parses('scene_parse/boxescamX0', rgb_camX0, gt_boxesX0_corners,torch.from_numpy(scores), tids, pix_T_cams[:, 0],info_text)
                if hyp.dataset_name == "carla":
                    summ_writer.summ_box_by_corners_parses('scene_parse/boxescamR', rgb_camtop, gt_boxescamXTop_corners,torch.from_numpy(scores), tids, pix_T_cams[:, 0],info_text)

            if not skip:
                summ_writer.summ_scalar('feat/perplexity',perplexity)
                summ_writer.summ_histogram('feat/encodings',e_indexes)            
            # print(e_indexes,self.info_dict['60'])
            # st()
            # print(torch.sum(self.quantizer.embeddings.weight))
            # print(torch.sum(self.featnet.net.final_feature.weight))
            # summ_writer.summ_embeddings('embeds', self.quantizer.embeddings.weight)
            # st()
            # summ_writer.summ_histogram('featnet_weights', self.featnet.net.final_feature.weight)
            # summ_writer.summ_histogram('embedings', self.quantizer.embeddings.weight)
            emb3D_e_R_object = quantized
            camX1_T_R = camXs_T_camRs[:,1]
            camX0_T_R = camXs_T_camRs[:,0]
            emb3D_R_non_updated = emb3D_R
            # st()
            # st()
            if not hyp.throw_away:
                emb3D_R = nlu.update_scene_with_objects(emb3D_R, emb3D_e_R_object ,gt_boxesRMem_end, scores)
            emb3D_e_X1 = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R)
            emb3D_e_X0 = utils_vox.apply_4x4_to_vox(camX0_T_R, emb3D_R)
            if hyp.gt_rotate_combinations:
                emb3D_R_best_rotated = nlu.update_scene_with_objects(emb3D_R, best_rotated_inputs ,gt_boxesRMem_end, scores)
                emb3D_e_X1_best_rotated = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R_best_rotated)
                emb3D_R_quantized_unrotated = nlu.update_scene_with_objects(emb3D_R, quantized_unrotated ,gt_boxesRMem_end, scores)
                emb3D_e_X1_quantized_unrotated = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R_quantized_unrotated)
                # st()
                # unpR_best_rotated = unp_visRs_eg_rotated_unstacked[best_rotations_index]
                # unp_trio = [unps_visX1s_eg.squeeze(1),unpR_best_rotated,unpR_best_rotated]

                # inp_best_quant_unp = torch.cat(,dim=2)
                # summ_writer.summ_rgb("selected_rotations_new2/unp_inp_best_quant",inp_best_quant_unp)
                # st()

            object_rgb = nlu.create_object_rgbs(rgb_camXs[:,0],gt_cornersX0_pix,scores)
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            rgb = object_rgb
            rgb = utils_improc.back2color(rgb)
            results['rgb'] = rgb
            results['classes'] = object_classes            
            results['valid3D'] = validR_combo_object.detach()
            # st()
            total_loss += hyp.quantize_loss_coef*loss_quant
            # st()
            # print("hello")


        if hyp.style_baseline:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_R = emb3D_e_R
            # st()
            emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2
            styles = emb3D_R_object.mean(dim=[2,3,4])
            contents = emb3D_R_object
            object_classes,filenames = nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            if (global_step % 300) == 1 or (global_step % 300) == 0:
                # print("Emptying \n")
                wrong = False
                st()
                try:
                    precision_style = float(self.tp_style) /self.all_style
                    precision_content = float(self.tp_content) /self.all_content
                except ZeroDivisionError:
                    wrong = True
                # st()
                if not wrong:
                    summ_writer.summ_scalar('precision/unsupervised_precision_style', precision_style)
                    summ_writer.summ_scalar('precision/unsupervised_precision_content', precision_content)
                    st()
                self.embed_list_style = defaultdict(lambda:[])
                self.embed_list_content = defaultdict(lambda:[])
                self.tp_style = 0
                self.all_style = 0
                self.tp_content = 0
                self.all_content = 0
                self.check = False
            elif not self.check and not nlu.check_fill_dict(self.embed_list_content,self.embed_list_style):
                print("Filling \n")
                for index,class_val in enumerate(object_classes):
                    # class_val = class_val[0]
                    if hyp.dataset_name == "clevr_vqa":
                        class_val_content, class_val_style = class_val.split("/")
                    else:
                        class_val_content, class_val_style = [class_val.split("/")[0],class_val.split("/")[0]]
                        # if class_val_content == "green_apple" or class_val_content == "appricot" or class_val_content == 'red_apple' or class_val_content == 'pear':
                        #     class_val_content = 'apple'
                        # if class_val_content == "small_banana" or class_val_content == "big_banana" or class_val_content == 'red_apple':
                        #     class_val_content = 'banana'
                        # if class_val_content == "tomato" or class_val_content == "avocado" or class_val_content == 'yellow_lemon' or class_val_content == 'green_lemon' or class_val_content ==  'red_peach':
                        #     class_val_content = 'lemon'                                
                        # if class_val_content == "green_grapes" or class_val_content == "red_grapes" or class_val_content == 'black_grapes':
                        #     class_val_content = 'grape'
                        # st()
                    # st()
                    print(len(self.embed_list_style.keys()),"style class",len(self.embed_list_content),"content class",self.embed_list_content.keys())
                    if len(self.embed_list_style[class_val_style]) < hyp.few_shot_nums:
                        self.embed_list_style[class_val_style].append(styles[index].squeeze())
                    if len(self.embed_list_content[class_val_content]) < hyp.few_shot_nums:
                        if hyp.avg_3d:
                            content_val = contents[index]
                            content_val = torch.mean(content_val.reshape([content_val.shape[1],-1]),dim=-1)
                            # st()
                            self.embed_list_content[class_val_content].append(content_val)
                        else:
                            self.embed_list_content[class_val_content].append(contents[index].reshape([-1]))
            else:
                self.check = True
                # print("Evaluating \n")
                # print(str(len(list(self.embed_list_style.keys())))+"\n")
                for key,val in self.embed_list_style.items():
                    if isinstance(val,type([])):
                        self.embed_list_style[key] = torch.mean(torch.stack(val,dim=0),dim=0)

                for key,val in self.embed_list_content.items():
                    if isinstance(val,type([])):
                        self.embed_list_content[key] = torch.mean(torch.stack(val,dim=0),dim=0)
                # st()
                    # st()
                for index,class_val in enumerate(object_classes):
                    class_val = class_val
                    if hyp.dataset_name == "clevr_vqa":
                        class_val_content, class_val_style = class_val.split("/")
                    else:
                        class_val_content, class_val_style = [class_val.split("/")[0],class_val.split("/")[0]]
                        # if class_val_content == "green_apple" or class_val_content == "appricot" or class_val_content == 'red_apple' or class_val_content == 'pear':
                        #     class_val_content = 'apple'
                        # if class_val_content == "small_banana" or class_val_content == "big_banana" or class_val_content == 'red_apple':
                        #     class_val_content = 'banana'
                        # if class_val_content == "tomato" or class_val_content == "avocado" or class_val_content == 'yellow_lemon' or class_val_content == 'green_lemon' or class_val_content ==  'red_peach':
                        #     class_val_content = 'lemon'                                
                        # if class_val_content == "green_grapes" or class_val_content == "red_grapes" or class_val_content == 'black_grapes':
                        #     class_val_content = 'grape'

                    
                    # style
                    style_val = styles[index].squeeze().unsqueeze(0)
                    embed_list_val_style = torch.stack(list(self.embed_list_style.values()),dim=0)
                    embed_list_key_style = list(self.embed_list_style.keys())
                    embed_list_val_style = utils_basic.l2_normalize(embed_list_val_style,dim=1).permute(1,0)
                    style_val = utils_basic.l2_normalize(style_val,dim=1)
                    scores_styles = torch.matmul(style_val,embed_list_val_style)
                    index_key = torch.argmax(scores_styles,dim=1).squeeze()
                    selected_class_style = embed_list_key_style[index_key]
                    self.styles_prediction[class_val_style].append(selected_class_style)
                    if class_val_style == selected_class_style:
                        self.tp_style += 1
                    self.all_style += 1
                    # content
                    # st()
                    if hyp.avg_3d:
                        content_val = contents[index]
                        content_val = torch.mean(content_val.reshape([content_val.shape[1],-1]),dim=-1).unsqueeze(0)
                    else:
                        content_val = contents[index].reshape([-1]).unsqueeze(0)

                    embed_list_val_content = torch.stack(list(self.embed_list_content.values()),dim=0)
                    embed_list_key_content = list(self.embed_list_content.keys())
                    embed_list_val_content = utils_basic.l2_normalize(embed_list_val_content,dim=1).permute(1,0)
                    content_val = utils_basic.l2_normalize(content_val,dim=1)
                    scores_content = torch.matmul(content_val,embed_list_val_content)
                    index_key = torch.argmax(scores_content,dim=1).squeeze()
                    selected_class_content = embed_list_key_content[index_key]
                    self.content_prediction[class_val_content].append(selected_class_content)
                    if class_val_content == selected_class_content:
                        self.tp_content += 1

                    self.all_content += 1



        if hyp.style_transfer:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_R = emb3D_e_R            


            emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2

            camX1_T_R = camXs_T_camRs[:,1]
            camX0_T_R = camXs_T_camRs[:,0]            
            assert hyp.B == 2
            assert emb3D_e_R_object.shape[0] == 2

            content_0 = emb3D_R_object[:1]
            style_0 = content_0

            content_1 = emb3D_R_object[1:]
            style_1 = content_1
            # st()
            if hyp.testset:
                sudo_content_0 = utils_geom.adin(content_0,style_1)
                sudo_content_1 = utils_geom.adin(content_1,style_0)                
                sudo_content_1 = utils_geom.adin(content_0,style_1)
                sudo_content_0 = utils_geom.adin(content_1,style_0)                
            else:
                sudo_content_1 = utils_geom.adin(content_0,style_1)
                sudo_content_0 = utils_geom.adin(content_1,style_0)
            content_obj = torch.cat([sudo_content_0,sudo_content_1],dim=0)
            emb3D_R = nlu.update_scene_with_objects(emb3D_R, content_obj ,gt_boxesRMem_end, scores)
            emb3D_e_X1 = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R)
            emb3D_e_X0 = utils_vox.apply_4x4_to_vox(camX0_T_R, emb3D_R)

        if hyp.voxel_quantize:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)            
            emb3D_R =  emb3D_e_R
            loss_quant, quantized, perplexity, encodings = self.quantizer(emb3D_R)
            emb3D_R = quantized
            camX1_T_R = camXs_T_camRs[:,1]
            emb3D_e_X1 = utils_vox.apply_4x4_to_vox(camX1_T_R, emb3D_R)
            total_loss += hyp.quantize_loss_coef*loss_quant
        start_time = time.time()

        if hyp.do_occ and hyp.occ_do_cheap:
            occX0_sup, freeX0_sup,_, freeXs = utils_vox.prep_occs_supervision(
                camX0_T_camXs,
                xyz_camXs,
                Z2,Y2,X2,
                agg=True)
            # st()

            summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))
            # st()
            if hyp.object_quantize or hyp.style_transfer:
                occ_loss, occX0s_pred_ = self.occnet(emb3D_e_X0,
                                                     occX0_sup,
                                                     freeX0_sup,
                                                     torch.max(validX0s[:,1:], dim=1)[0],
                                                     summ_writer)

            else:
                occ_loss, occX0s_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                     occX0_sup,
                                                     freeX0_sup,
                                                     torch.max(validX0s[:,1:], dim=1)[0],
                                                     summ_writer)
            occX0s_pred = __u(occX0s_pred_)
            total_loss += occ_loss
            if hyp.do_empty:
                if self.is_empty_occ_generated == False:
                    self.is_empty_occ_generated  = True
                    empty_occX0_sup, empty_freeX0_sup,_, empty_freeXs = utils_vox.prep_occs_supervision(
                        camX0_T_camXs,
                        empty_xyz_camXs,
                        Z2,Y2,X2,
                        agg=True)
                    
                    summ_writer.summ_occ('occ_sup/empty_occ_sup', empty_occX0_sup)
                    summ_writer.summ_occ('occ_sup/empty_free_sup', empty_freeX0_sup)
                    summ_writer.summ_occs('occ_sup/empty_freeXs_sup', torch.unbind(empty_freeXs, dim=1))
                    summ_writer.summ_occs('occ_sup/empty_occXs_sup', torch.unbind(empty_occXs_half, dim=1))

                    empty_occ_loss, empty_occX0s_pred_ = self.occnet(torch.mean(empty_featX0s[:,1:], dim=1),
                                                        empty_occX0_sup,
                                                        empty_freeX0_sup,
                                                        torch.max(empty_validX0s[:,1:], dim=1)[0],
                                                        summ_writer,
                                                        prefix="_empty",
                                                        log_summ = True)

                    empty_occX0s_pred = __u(empty_occX0s_pred_)
                    self.empty_occ = empty_occX0s_pred_
                occ_subtracted_e = occX0s_pred_ - self.empty_occ
                occ_subtracted_e[occ_subtracted_e>0]=1                
                summ_writer.summ_occ('occ_sup/occ_subtracted_e', occ_subtracted_e)
                # self.bounding_box_generator.subtract_point_clouds(occX0s_pred_.cpu().numpy(), empty_occX0s_pred_.cpu().numpy(), vis=True)
            if hyp.profile_time:                
                print("occ time",time.time()-start_time)
        
        start_time = time.time()
        if hyp.do_view and not hyp.store_obj:
            assert(hyp.do_feat)
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            if hyp.debug_match and hyp.debug_aug:
                view_to_take = camX1_T_camRs[:1]
                camX1_T_camRs[1:] = view_to_take
                view_to_take_2 = camX0_T_camXs[:1]
                camX0_T_camXs[1:] = view_to_take_2
                emb3D_X1_aug = utils_vox.apply_4x4_to_vox(camX1_T_camRs, emb3D_R_aug)
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_X1_aug,
                    hyp.view_depth, PH, PW)   
                # st()
                feat_projX1 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX1_T_camRs, emb3D_R_aug,
                    hyp.view_depth, PH, PW)                                        
            elif hyp.debug_aug and feed['set_name'] == "train":
                hyp.lr == 0.0
                if hyp.debug_add or hyp.aug_object_ent:
                    emb3D_X1_aug_old = utils_vox.apply_4x4_to_vox(camX1_T_camRs, emb3D_R_aug_old)
                    feat_projX00_old = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_X1_aug_old,
                        hyp.view_depth, PH, PW)                    
                
                emb3D_X1_aug = utils_vox.apply_4x4_to_vox(camX1_T_camRs, emb3D_R_aug)
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_X1_aug,
                    hyp.view_depth, PH, PW)

            elif hyp.object_quantize or hyp.style_transfer or hyp.do_munit:
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1, # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)      
                
                if hyp.do_munit:
                    feat_projX00_og  = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1_og, # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)      

                    # only for checking the style
                    styled_feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,0], camX0_T_camXs[:,1], styled_emb3D_e_X1, # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)

                    if hyp.cycle_style_view_loss:
                        styled_feat_projX00_cycle = utils_vox.apply_pixX_T_memR_to_voxR(
                            projpix_T_cams[:,0], camX0_T_camXs[:,1], styled_emb3D_e_X1_cycle, # use feat1 to predict rgb0
                            hyp.view_depth, PH, PW)                         

                if hyp.gt_rotate_combinations:
                    feat_projX00_best_rotated = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1_best_rotated, # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)                      
                    feat_projX00_quantized_unrotated = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1_quantized_unrotated, # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)                                          
            elif hyp.voxel_quantize:
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1, # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)
            else:        
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], featXs[:,1], # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)
            rgb_X00 = utils_basic.downsample(rgb_camXs[:,0], 2)
            rgb_X01 = utils_basic.downsample(rgb_camXs[:,1], 2)
            valid_X00 = utils_basic.downsample(valid_camXs[:,0], 2)
            # decode the perspective volume into an image
            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                valid_X00,
                summ_writer,"rgb")
            if hyp.debug_match:
                _, rgb_e_v2, _ = self.viewnet(
                    feat_projX1,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"rgb_v2")                 
                # st()
            # st()
            if hyp.debug_match:
                summ_writer.summ_rgb("aug/rgb_og",rgb_e[:1])
                summ_writer.summ_rgb("aug/rgb_fake",rgb_e[1:])
                summ_writer.summ_rgb("aug/rgb_og_v2",rgb_e_v2[:1])
                summ_writer.summ_rgb("aug/rgb_fake_v2",rgb_e_v2[1:])
            if (hyp.debug_add or hyp.aug_object_ent) and hyp.debug_aug and feed['set_name'] == "train":
                _, rgb_e_old, emb2D_e = self.viewnet(
                feat_projX00_old,
                rgb_X00,
                valid_X00,
                summ_writer,"rgb_old")
            if hyp.aug_object_ent and hyp.debug_aug and feed['set_name'] == "train":
                # all_aug_bboxesX0corners all_aug_score
                # all_gt_boxesX0_corners_aug all_gt_scores
                # st()
                summ_writer.summ_rgb("aug_boxes/rgb_old", rgb_e_old)
                summ_writer.summ_box_by_corners('aug_boxes/gt_boxescamX0', torch.nn.functional.interpolate(rgb_e,size=[hyp.H,hyp.W]), all_aug_bboxesX0corners, torch.from_numpy(all_aug_score), tids, pix_T_cams[:, 0])
                # st()
                # st()
                # print('hello')
            # if hyp.debug and hyp.do_munit:
            if hyp.do_munit:
                _, rgb_e, emb2D_e = self.viewnet(
                    feat_projX00_og,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"rgb_og")                
            if hyp.do_munit:
                styled_view_loss, styled_rgb_e, styled_emb2D_e = self.viewnet(
                    styled_feat_projX00,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"recon_style")
                if hyp.cycle_style_view_loss:
                    styled_view_loss_cycle, styled_rgb_e_cycle, styled_emb2D_e_cycle = self.viewnet(
                        styled_feat_projX00_cycle,
                        rgb_X00,
                        valid_X00,
                        summ_writer,"recon_style_cycle")                    

                # gt_rgx_x00 = torch.cat(torch.unbind(rgb_X00, dim=0), dim=-1)
                # imsave(f"dump/{global_step}.png",gt_rgx_x00.permute(1,2,0).cpu().numpy())
                # recon_rgb = torch.cat(torch.unbind(rgb_e, dim=0), dim=-1)
                # styled_rgb = torch.cat(torch.unbind(styled_rgb_e, dim=0), dim=-1)
                # complete_vis = torch.cat([gt_rgx_x00, recon_rgb, styled_rgb], dim=1)
                rgb_input_1 = torch.cat([rgb_X01[1],rgb_X01[0],styled_rgb_e[0]],dim=2)
                rgb_input_2 = torch.cat([rgb_X01[0],rgb_X01[1],styled_rgb_e[1]],dim=2)
                complete_vis = torch.cat([rgb_input_1,rgb_input_2],dim=1)        
                summ_writer.summ_rgb('munit/munit_recons_vis', complete_vis.unsqueeze(0))
                if hyp.save_rgb:
                    # imsave(f'dump/tmp.png',gt_rgx_x00.permute(1,2,0).cpu())
                    # st()
                    for index,rgb in enumerate([rgb_input_1,rgb_input_2]):
                        imsave(f'dump/{global_step}_{index}.png',rgb.permute(1,2,0).cpu())
                    # rgb_together = torch.cat([rgb_input,rgb_X00,rgb_e],dim=3)
                    # st()
                    # for index,rgb in enumerate(rgb_together):


            if hyp.save_rgb and False:
                rgb_input = torch.cat([rgb_X01[1:],rgb_X01[:1]],dim=0)
                rgb_together = torch.cat([rgb_input,rgb_X00,rgb_e],dim=3)
                # st()
                for index,rgb in enumerate(rgb_together):
                    imsave(f'dump/{global_step}_{index}.png',rgb.permute(1,2,0).cpu())

            if hyp.obj_multiview:
                projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))
                for i in range(hyp.S):
                    rgb_Xi = utils_basic.downsample(rgb_camXs[:,i], 2)
                    valid_Xi = utils_basic.downsample(valid_camXs[:,i], 2)                        
                    gt_boxesXi_corners = __ub(utils_geom.apply_4x4(camXs_T_camRs[:,i], __pb(gt_boxesR_corners)))
                    gt_cornersXi_pix = __ub(utils_geom.apply_pix_T_cam(projpix_T_cams[:,i], __pb(gt_boxesXi_corners)))
                    # st()
                    feat_projXi = utils_vox.apply_pixX_T_memR_to_voxR(
                        projpix_T_cams[:,i], utils_geom.eye_4x4(hyp.B), featXs[:,i], # use feat1 to predict rgb0
                        hyp.view_depth, PH, PW)
                    _, rgb_e_i, emb2D_e = self.viewnet(
                        feat_projXi,
                        rgb_Xi,
                        valid_Xi,
                        summ_writer,f"rgb_{i}")
                    object_rgb_Xi_pred = nlu.create_object_rgbs(rgb_e_i,gt_cornersXi_pix,scores)
                    summ_writer.summ_rgb(f"scene_parse/object_q_rpred_{i}",object_rgb_Xi_pred)
                    object_rgb_Xi_gt = nlu.create_object_rgbs(rgb_Xi,gt_cornersXi_pix,scores)
                    summ_writer.summ_rgb(f"scene_parse/object_q_rgt_{i}",object_rgb_Xi_gt)
            if hyp.gt_rotate_combinations and hyp.object_quantize:
                _, rgb_best, _ = self.viewnet(
                    feat_projX00_best_rotated,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"rgb_best_rotated")                                
                _, rgb_quant, _ = self.viewnet(
                    feat_projX00_quantized_unrotated,
                    rgb_X00,
                    valid_X00,
                    summ_writer,"rgb_quant_unrotated")                
                inp_best_quant = torch.cat([rgb_X00,rgb_best,rgb_quant],dim=2)
                # unp_trio = [torch.nn.functional.interpolate(i,[hyp.PH,hyp.PW]) for i in unp_trio]
                # unp_trio = torch.cat(unp_trio,dim=2)
                # inp_best_quant = torch.cat([inp_best_quant,unp_trio])
                # st()
                summ_writer.summ_rgb("selected_rotations/inp_best_quant",inp_best_quant)
            
            if not hyp.do_munit:
                total_loss += view_loss
            else:
                if hyp.basic_view_loss:
                    total_loss += view_loss
                if hyp.style_view_loss:
                    total_loss += styled_view_loss
                if hyp.cycle_style_view_loss:
                    total_loss += styled_view_loss_cycle

            
            if hyp.profile_time:
                print("view time",time.time()-start_time)
        if hyp.do_render:
            assert(hyp.do_feat)
            # we warped the features into the canonical view
            # now we resample to the target view and decode
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))

            assert(S==2) # else we should warp each feat in 1:
            feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], camX0_T_camXs[:,1], featXs[:,1], # use feat1 to predict rgb0
                hyp.view_depth, PH, PW)
            # feat_projX00 is B x hyp.feat_dim x hyp.view_depth x PH x PW
            occ_pred_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], camX0_T_camXs[:,0], occX0s_pred[:,0]*torch.max(validX0s[:,1:], dim=1)[0], # note occX0s already comes from feat1
                hyp.view_depth, PH, PW)
            occ_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                projpix_T_cams[:,0], camX0_T_camXs[:,0], occX0s_half[:,0],
                hyp.view_depth, PH, PW)
            # occ_projX00 is B x 1 x hyp.view_depth x PH x PW
            
            rgb_X00 = utils_basic.downsample(rgb_camXs[:,0], 2)
            valid_X00 = utils_basic.downsample(valid_camXs[:,0], 2)

            # decode the perspective volume into an image
            render_loss, rgb_e, emb2D_e = self.rendernet(
                feat_projX00,
                occ_pred_projX00,
                rgb_X00,
                valid_X00,
                summ_writer)
            total_loss += render_loss
        if hyp.do_emb2D:
            assert(hyp.do_view)
            # create an embedding image, representing the bottom-up 2D feature tensor
            emb2D_g = self.embnet2D_encoder(rgb_camXs[:,0])
            emb_loss_2D, emb2D_g = self.embnet2D(
                emb2D_g,
                emb2D_e,
                valid_camXs[:,0],
                summ_writer)
            total_loss += emb_loss_2D
        
        if hyp.do_emb3D:
            if hyp.emb3D_o:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_e_R_object,emb3D_g_R_object,vis3D_g_R,validR_combo_object = nlu.create_object_tensors([emb3D_e_R,emb3D_g_R],[vis3D_g_R,validR_combo],gt_boxesRMem_end,scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
                object_rgb = nlu.create_object_rgbs(rgb_camXs[:,0],gt_cornersX0_pix,scores)
                object_classes,filenames = nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            emb_loss_3D = self.embnet3D(
                emb3D_e_R_object,
                emb3D_g_R_object,
                vis3D_g_R,
                summ_writer)
            rgb = object_rgb
            total_loss += emb_loss_3D

        if hyp.offline_cluster or hyp.offline_cluster_eval:
            emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
            emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
            emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            results['emb3D_e'] = emb3D_e_R_object
            results['emb3D_g'] = emb3D_g_R_object
            results['classes'] = object_classes
        # st()
        if (hyp.moc or (hyp.do_eval_recall) or hyp.break_constraint) and not hyp.object_quantize:
            if hyp.moc_2d:
                emb2D_g = self.embnet2D_encoder(rgb_camXs[:,0])
                results['emb2D_e'] = emb2D_e
                results['emb2D_g'] = emb2D_g
            if not hyp.emb3D_o and hyp.eval_recall_o or hyp.online_cluster:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)

                if hyp.debug_eval_recall_o:
                    emb3D_e_R = torch.mean(occRs_half[:,1:], dim=1)
                    emb3D_g_R = occRs_half[:,0]

                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])

                if hyp.debug_eval_recall_o:
                    summ_writer.summ_occs('debug_eval_recall/emb3D_e_occR', [emb3D_e_R_object])
                    summ_writer.summ_occs('debug_eval_recall/emb3D_g_occR', [emb3D_g_R_object])

                object_rgb = nlu.create_object_rgbs(rgb_camXs[:,0],gt_cornersX0_pix,scores)
                object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
                rgb = object_rgb
                rgb = utils_improc.back2color(rgb)

            if hyp.eval_recall_o or hyp.online_cluster:
                rgb_vis3D = None
                if hyp.hard_vis or hyp.online_cluster:
                    unpRs_comb = torch.mean(unpRs,dim=1,keepdim=False)
                    occRs_comb = torch.max(occRs,dim=1,keepdim=False).values
                    unpRs_eg, _ = nlu.create_object_tensors([unpRs_comb], None, gt_boxesRUnp_end, scores,[BOX_SIZE*2,BOX_SIZE*2,BOX_SIZE*2])
                    occRs_eg, _ = nlu.create_object_tensors([occRs_comb], None, gt_boxesRUnp_end, scores,[BOX_SIZE*2,BOX_SIZE*2,BOX_SIZE*2],occs=True)
                    unps_visRs_eg = utils_improc.get_unps_vis(unpRs_eg.unsqueeze(1), occRs_eg.unsqueeze(1))
                    unp_visRs_eg = torch.mean(unps_visRs_eg, dim=1)
                    if hyp.online_cluster:
                        occRs_obj, _ = nlu.create_object_tensors([occRs_comb], None, gt_boxesRUnp_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE],occs=True)
                        summ_writer.summ_occ('3D_inputs/occR_cropped', occRs_obj)
                        summ_writer.summ_unp('3D_inputs/unpR_cropped', unpRs_eg, occRs_eg)
                        emb3D_R_mean_crop = torch.mean(torch.stack([emb3D_e_R_object,emb3D_g_R_object],dim=0),dim=0)
                        emb3D_R_mean_crop_flat = emb3D_R_mean_crop.reshape([hyp.B,hyp.feat_dim,-1])
                        emb3D_R_mean_crop_flat = emb3D_R_mean_crop_flat.permute(0,2,1).cpu().numpy()
                        occRs_obj_flat_mask = occRs_obj.reshape([hyp.B,-1]).to(torch.bool).cpu().numpy()
                        for index,emb_i in enumerate(emb3D_R_mean_crop_flat):
                            occRs_i_mask = occRs_obj_flat_mask[index]
                            emb_i_filtered = emb_i[occRs_i_mask]
                            if self.voxel_queue.is_full():
                                voxels = self.voxel_queue.fetch()
                                voxels = np.stack(voxels)
                                self.kmeans.partial_fit(voxels[::10])
                                self.voxel_queue.update(emb_i_filtered[::5])
                            else:
                                self.voxel_queue.update(emb_i_filtered[::5])
                        # predictions
                        if hyp.online_cluster_eval and summ_writer.save_this and self.voxel_queue.is_full():
                            index = 0 
                            embR_flat_b = emb3D_R_mean_crop_flat[index]
                            occR_obj_flat_mask_b = occRs_obj_flat_mask[index]

                            clusters_b = self.kmeans.predict(embR_flat_b)
                            clusters_b[np.logical_not(occR_obj_flat_mask_b)] = -1
                            clusters_b_masked = clusters_b[occR_obj_flat_mask_b]
                            embR_flat_b_masked = embR_flat_b[occR_obj_flat_mask_b]
                            summ_writer.summ_histogram(f'emb_B{index}',
                                                       clusters_b_masked)
                            # summ_writer.summ_embeddings(f'emb_B{index}',
                            #                             embR_flat_b_masked,
                            #                             clusters_b_masked)
                            centers = self.kmeans.cluster_centers_
                            for m in range(hyp.object_quantize_dictsize):
                                mode,count = scipy.stats.mode(clusters_b_masked)
                                print(f'mode {m} -- {count[0]}')
                                if count[0] == 0: break
                                center = centers[mode]
                                roi = clusters_b == mode
                                dist = np.linalg.norm(embR_flat_b[roi]-center,
                                                      axis=1)
                                summ_writer.summ_histogram(f'dist/emb_B{index}M{m+1}',
                                                           dist)
                                roi = np.reshape(roi,[1,1,hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE])
                                roi = torch.from_numpy(roi).cuda().to(torch.float32)
                                summ_writer.summ_occ(f'clusters/emb_B{index}M{m+1}',roi,
                                                     reduce_axes=[2,3])
                                ind = clusters_b_masked != mode
                                clusters_b_masked = clusters_b_masked[ind]
                                if clusters_b_masked.shape[0] == 0: break
                        else:
                            index = 0
                            roi = np.zeros([1,1,hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE])
                            roi = torch.from_numpy(roi).cuda().to(torch.float32)
                            for m in range(hyp.object_quantize_dictsize):
                                summ_writer.summ_occ(f'clusters/emb_B{index}M{m+1}',roi,
                                                     reduce_axes=[2,3])
                        # print("")
                    results['visual3D'] = unp_visRs_eg
                results['classes'] = object_classes
            else:
                rgb_vis3D = utils_improc.back2color(utils_basic.reduce_masked_mean(unpRs,occRs.repeat(1, 1, 3, 1, 1, 1),dim=1))
                rgb = rgb_camXs[:, 0]
                rgb = torch.nn.functional.interpolate(rgb, size=[hyp.PH*2, hyp.PW*2], mode='bilinear')            
                rgb = utils_improc.back2color(rgb)
            try:
                results['emb3D_e'] = emb3D_e_R_object
                results['emb3D_g'] = emb3D_g_R_object
                results['rgb'] = rgb

                if validR_combo_object is not None:
                    validR_combo_object = validR_combo_object.detach()
                results['valid3D'] = validR_combo_object
            except Exception:
                pass
        if not hyp.moc:
            summ_writer.summ_scalar('loss', total_loss.cpu().item())

        if hyp.save_embed_tsne:
            for index,class_val in enumerate(object_classes):
                class_val_content, class_val_style = class_val.split("/")
                style_val =  styles[index].squeeze().unsqueeze(0)
                self.cluster_pool.update(style_val, [class_val_style])
                print(self.cluster_pool.num)
            
            if self.cluster_pool.is_full():
                embeds,classes = self.cluster_pool.fetch()
                with open("offline_cluster" + '/%st.txt' % 'classes', 'w') as f:
                    for index,embed in enumerate(classes):
                        class_val = classes[index]
                        f.write("%s\n" % class_val)
                f.close()
                with open("offline_cluster" + '/%st.txt' % 'embeddings', 'w') as f:
                    for index,embed in enumerate(embeds):
                        # embed = utils_basic.l2_normalize(embed,dim=0)
                        print("writing {} embed".format(index))
                        embed_l_s = [str(i) for i in embed.tolist()]
                        embed_str = '\t'.join(embed_l_s)
                        f.write("%s\n" % embed_str)
                f.close()
                st()



        return total_loss, results

