import torch
import torch.nn as nn
import hyperparams as hyp
import cross_corr
import numpy as np
import imageio
import os
import json
from model_base import Model
from nets.featnet import FeatNet
from nets.occnet import OccNet
from nets.viewnet import ViewNet
from nets.rendernet import RenderNet
from nets.munitnet import MunitNet,MunitNet_Simple
from collections import defaultdict
import torch.nn.functional as F
from scipy.misc import imsave
from collections import defaultdict
from os.path import join
import time
import random
import glob
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
import utils_eval
import sklearn
from DoublePool import SinglePool
import torchvision.models as models
from lib_classes import Nel_Utils as nlu
import copy

np.set_printoptions(precision=2)
np.random.seed(0)


class CLEVR_STA(Model):

    def infer(self):
        print("------ BUILDING INFERENCE GRAPH ------")
        self.model = ClevrStaNet()
        if hyp.do_freeze_feat:
            self.model.featnet.eval()
            self.set_requires_grad(self.model.featnet, False)


class ClevrStaNet(nn.Module):
    def __init__(self):
        super(ClevrStaNet, self).__init__()
        self.device = "cuda"
        self.list_of_classes = []
        

        self.minclasses = 3            

        # self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE,hyp.BOX_SIZE,hyp.BOX_SIZE)

        self.info_dict = defaultdict(lambda:[])
            
            
        self.embed_list_style = defaultdict(lambda:[])
        self.embed_list_content = defaultdict(lambda:[])

        if hyp.do_feat:
            self.featnet = FeatNet()
        if hyp.do_occ or (hyp.remove_air and hyp.aug_det):
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()
        if hyp.do_render:
            self.rendernet = RenderNet()

        
        if hyp.do_munit:
            if hyp.simple_adaingen:
                self.munitnet = MunitNet_Simple().cuda()
            else:
                self.munitnet = MunitNet().cuda()

        self.is_empty_occ_generated = False
        self.avg_ap = []
        self.avg_precision = []                    
        self.tp_style = 0
        self.all_style = 0
        self.tp_content = 0
        self.all_content = 0        

        self.max_content = None
        self.min_content = None 
        self.max_style = None 
        self.min_style = None
        self.styles_prediction = defaultdict(lambda:[])
        self.content_prediction = defaultdict(lambda:[])


    def load_config(self,exp_name):
        path = os.path.join('experiments', exp_name, 'config.json')
        with open(path) as file:
            config = json.load(file)
        assert config['name']==exp_name
        return config


    def forward(self, feed):
        results = dict()

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

        rgb_camXs = feed["rgb_camXs_raw"]
        pix_T_cams = feed["pix_T_cams_raw"]
        camRs_T_origin = feed["camR_T_origin_raw"]
        origin_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_origin)))
        origin_T_camXs = feed["origin_T_camXs_raw"]
        camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
        camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
        camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
        camX0_T_camRs = camXs_T_camRs[:,0]
        camX1_T_camRs = camXs_T_camRs[:,1]
        
        camR_T_camX0  = utils_geom.safe_inverse(camX0_T_camRs)

        xyz_camXs = feed["xyz_camXs_raw"]
        depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
        dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))


        xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
        xyz_camX0s = __u(utils_geom.apply_4x4(__p(camX0_T_camXs), __p(xyz_camXs)))

        occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
        
        occXs_to_Rs = utils_vox.apply_4x4s_to_voxs(camRs_T_camXs, occXs) 
        occXs_to_Rs_45 = cross_corr.rotate_tensor_along_y_axis(occXs_to_Rs, 45)
        occXs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z2, Y2, X2))
        occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
        occX0s_half = __u(utils_vox.voxelize_xyz(__p(xyz_camX0s), Z2, Y2, X2))

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
            
         
        dense_xyz_camRs_ = utils_geom.apply_4x4(__p(camRs_T_camXs), dense_xyz_camXs_)
        inbound_camXs_ = utils_vox.get_inbounds(dense_xyz_camRs_, Z, Y, X).float()
        inbound_camXs_ = torch.reshape(inbound_camXs_, [B*S, 1, H, W])
        
        depth_camXs = __u(depth_camXs_)
        valid_camXs = __u(valid_camXs_) * __u(inbound_camXs_)


        summ_writer.summ_oneds('2D_inputs/depth_camXs', torch.unbind(depth_camXs, dim=1),maxdepth=21.0)
        summ_writer.summ_oneds('2D_inputs/valid_camXs', torch.unbind(valid_camXs, dim=1))
        summ_writer.summ_rgbs('2D_inputs/rgb_camXs', torch.unbind(rgb_camXs, dim=1))
        summ_writer.summ_occs('3D_inputs/occXs', torch.unbind(occXs, dim=1))
        summ_writer.summ_unps('3D_inputs/unpXs', torch.unbind(unpXs, dim=1), torch.unbind(occXs, dim=1))

        occRs = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z, Y, X))
        
        if hyp.do_eval_boxes:
            if hyp.dataset_name =="clevr_vqa":
                gt_boxes_origin_corners = feed['gt_box']
                gt_scores_origin = feed['gt_scores'].detach().cpu().numpy()
                classes = feed['classes']
                scores = gt_scores_origin
                tree_seq_filename = feed['tree_seq_filename']
                gt_boxes_origin = nlu.get_ends_of_corner(gt_boxes_origin_corners)
                gt_boxes_origin_end = torch.reshape(gt_boxes_origin,[hyp.B,hyp.N,2,3])
                gt_boxes_origin_theta = nlu.get_alignedboxes2thetaformat(gt_boxes_origin_end)
                gt_boxes_origin_corners = utils_geom.transform_boxes_to_corners(gt_boxes_origin_theta)
                gt_boxesR_corners = __ub(utils_geom.apply_4x4(camRs_T_origin[:,0], __pb(gt_boxes_origin_corners)))
                gt_boxesR_theta = utils_geom.transform_corners_to_boxes(gt_boxesR_corners)
                gt_boxesR_end = nlu.get_ends_of_corner(gt_boxesR_corners)

            else:
                tree_seq_filename = feed['tree_seq_filename']
                tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename if i != "invalid_tree"]
                invalid_tree_filenames = [join(hyp.root_dataset,i) for i in tree_seq_filename if i == "invalid_tree"]
                num_empty = len(invalid_tree_filenames)                    
                trees = [pickle.load(open(i,"rb")) for i in tree_filenames]

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

            class_names_ex_1 = "_".join(classes[0])
            summ_writer.summ_text('eval_boxes/class_names', class_names_ex_1)
            
            gt_boxesRMem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z2,Y2,X2))
            gt_boxesRMem_end = nlu.get_ends_of_corner(gt_boxesRMem_corners)

            gt_boxesRMem_theta = utils_geom.transform_corners_to_boxes(gt_boxesRMem_corners)
            gt_boxesRUnp_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesR_corners),Z,Y,X))
            gt_boxesRUnp_end = nlu.get_ends_of_corner(gt_boxesRUnp_corners)
            
            gt_boxesX0_corners = __ub(utils_geom.apply_4x4(camX0_T_camRs, __pb(gt_boxesR_corners)))
            gt_boxesX0Mem_corners = __ub(utils_vox.Ref2Mem(__pb(gt_boxesX0_corners),Z2,Y2,X2))

            gt_boxesX0Mem_theta = utils_geom.transform_corners_to_boxes(gt_boxesX0Mem_corners)
            
            gt_boxesX0Mem_end = nlu.get_ends_of_corner(gt_boxesX0Mem_corners)
            gt_boxesX0_end = nlu.get_ends_of_corner(gt_boxesX0_corners)

            gt_cornersX0_pix = __ub(utils_geom.apply_pix_T_cam(pix_T_cams[:,0], __pb(gt_boxesX0_corners)))

            rgb_camX0 = rgb_camXs[:,0]
            rgb_camX1 = rgb_camXs[:,1]

            summ_writer.summ_box_by_corners('eval_boxes/gt_boxescamX0', rgb_camX0, gt_boxesX0_corners, torch.from_numpy(scores), tids, pix_T_cams[:, 0])
            unps_vis = utils_improc.get_unps_vis(unpX0s_half, occX0s_half)
            unp_vis = torch.mean(unps_vis, dim=1)
            unps_visRs = utils_improc.get_unps_vis(unpRs_half, occRs_half)
            unp_visRs = torch.mean(unps_visRs, dim=1)
            unps_visRs_full = utils_improc.get_unps_vis(unpRs, occRs)
            unp_visRs_full = torch.mean(unps_visRs_full, dim=1)
            summ_writer.summ_box_mem_on_unp('eval_boxes/gt_boxesR_mem', unp_visRs , gt_boxesRMem_end, scores ,tids)
            
            unpX0s_half = torch.mean(unpX0s_half, dim=1)
            unpX0s_half = nlu.zero_out(unpX0s_half,gt_boxesX0Mem_end,scores)

            occX0s_half = torch.mean(occX0s_half, dim=1)
            occX0s_half = nlu.zero_out(occX0s_half,gt_boxesX0Mem_end,scores)            

            summ_writer.summ_unp('3D_inputs/unpX0s', unpX0s_half, occX0s_half)

        if hyp.do_feat:
            featXs_input = torch.cat([occXs, occXs*unpXs], dim=2)
            featXs_input_ = __p(featXs_input)
            
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

            featXs = __u(featXs_)
            _featX00 = featXs[:,0:1]
            _featX01 = utils_vox.apply_4x4s_to_voxs(camX0_T_camXs[:,1:], featXs[:,1:])
            featX0s = torch.cat([_featX00, _featX01], dim=1)

            emb3D_e = torch.mean(featX0s[:,1:], dim=1) 
            vis3D_e_R = torch.max(visRs[:,1:], dim=1)[0]
            emb3D_g = featX0s[:,0] 
            vis3D_g_R = visRs[:,0] 
            validR_combo = torch.min(validRs,dim=1).values


            summ_writer.summ_feats('3D_feats/featXs_input', torch.unbind(featXs_input, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featXs_output', torch.unbind(featXs, dim=1), valids=torch.unbind(validXs, dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/featX0s_output', torch.unbind(featX0s, dim=1), valids=torch.unbind(torch.ones_like(validRs), dim=1), pca=True)
            summ_writer.summ_feats('3D_feats/validRs', torch.unbind(validRs, dim=1), pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_e_R', vis3D_e_R, pca=False)
            summ_writer.summ_feat('3D_feats/vis3D_g_R', vis3D_g_R, pca=False)


        if hyp.do_munit:
            object_classes,filenames= nlu.create_object_classes(classes,[tree_seq_filename,tree_seq_filename],scores)
            if hyp.do_munit_fewshot:
                emb3D_e_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_e)
                emb3D_g_R = utils_vox.apply_4x4_to_vox(camR_T_camX0, emb3D_g)
                emb3D_R = emb3D_e_R
                emb3D_e_R_object, emb3D_g_R_object, validR_combo_object = nlu.create_object_tensors([emb3D_e_R, emb3D_g_R], [validR_combo], gt_boxesRMem_end, scores,[BOX_SIZE,BOX_SIZE,BOX_SIZE])
                emb3D_R_object = (emb3D_e_R_object + emb3D_g_R_object)/2
                content,style = self.munitnet.net.gen_a.encode(emb3D_R_object)
                objects_taken,_ = self.munitnet.net.gen_a.decode(content, style)
                styles = style
                contents = content
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

            if hyp.run_few_shot_on_munit: 
                if (global_step % 300) == 1 or (global_step % 300) == 0:
                    wrong = False
                    try:
                        precision_style = float(self.tp_style) /self.all_style
                        precision_content = float(self.tp_content) /self.all_content
                    except ZeroDivisionError:
                        wrong = True

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

                        if hyp.dataset_name == "clevr_vqa":
                            class_val_content, class_val_style = class_val.split("/")
                        else:
                            class_val_content, class_val_style = [class_val.split("/")[0],class_val.split("/")[0]]

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
            # st()
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

            summ_writer.summ_feat(f'aug_feat/og', emb3D_R)
            summ_writer.summ_feat(f'aug_feat/og_gen', recon_emb3D_R)
            summ_writer.summ_feat(f'aug_feat/og_aug_diff', emb3D_R_aug_diff)

            if hyp.cycle_style_view_loss:
                sudo_input_obj_cycle = torch.cat([sudo_input_0_cycle,sudo_input_1_cycle],dim=0)
                styled_emb3D_R_cycle = nlu.update_scene_with_objects(emb3D_R, sudo_input_obj_cycle, gt_boxesRMem_end, scores)

                styled_emb3D_e_X0_cycle = utils_vox.apply_4x4_to_vox(camX0_T_R, styled_emb3D_R_cycle)
                styled_emb3D_e_X1_cycle = utils_vox.apply_4x4_to_vox(camX1_T_R, styled_emb3D_R_cycle)
            summ_writer.summ_scalar('munit_loss', munit_loss.cpu().item())
            total_loss += munit_loss

        if hyp.do_occ and hyp.occ_do_cheap:
            occX0_sup, freeX0_sup,_, freeXs = utils_vox.prep_occs_supervision(
                camX0_T_camXs,
                xyz_camXs,
                Z2,Y2,X2,
                agg=True)

            summ_writer.summ_occ('occ_sup/occ_sup', occX0_sup)
            summ_writer.summ_occ('occ_sup/free_sup', freeX0_sup)
            summ_writer.summ_occs('occ_sup/freeXs_sup', torch.unbind(freeXs, dim=1))
            summ_writer.summ_occs('occ_sup/occXs_sup', torch.unbind(occXs_half, dim=1))

            occ_loss, occX0s_pred_ = self.occnet(torch.mean(featX0s[:,1:], dim=1),
                                                 occX0_sup,
                                                 freeX0_sup,
                                                 torch.max(validX0s[:,1:], dim=1)[0],
                                                 summ_writer)
            occX0s_pred = __u(occX0s_pred_)
            total_loss += occ_loss
        
        
        if hyp.do_view:
            assert(hyp.do_feat)
            PH, PW = hyp.PH, hyp.PW
            sy = float(PH)/float(hyp.H)
            sx = float(PW)/float(hyp.W)
            assert(sx==0.5) # else we need a fancier downsampler
            assert(sy==0.5)
            projpix_T_cams = __u(utils_geom.scale_intrinsics(__p(pix_T_cams), sx, sy))
            st()

            if hyp.do_munit:
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], emb3D_e_X1, # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)      
                
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

            else:        
                feat_projX00 = utils_vox.apply_pixX_T_memR_to_voxR(
                    projpix_T_cams[:,0], camX0_T_camXs[:,1], featXs[:,1], # use feat1 to predict rgb0
                    hyp.view_depth, PH, PW)
            rgb_X00 = utils_basic.downsample(rgb_camXs[:,0], 2)
            rgb_X01 = utils_basic.downsample(rgb_camXs[:,1], 2)
            valid_X00 = utils_basic.downsample(valid_camXs[:,0], 2)

            view_loss, rgb_e, emb2D_e = self.viewnet(
                feat_projX00,
                rgb_X00,
                valid_X00,
                summ_writer,"rgb")


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

                rgb_input_1 = torch.cat([rgb_X01[1],rgb_X01[0],styled_rgb_e[0]],dim=2)
                rgb_input_2 = torch.cat([rgb_X01[0],rgb_X01[1],styled_rgb_e[1]],dim=2)
                complete_vis = torch.cat([rgb_input_1,rgb_input_2],dim=1)        
                summ_writer.summ_rgb('munit/munit_recons_vis', complete_vis.unsqueeze(0))

            
            if not hyp.do_munit:
                total_loss += view_loss
            else:
                if hyp.basic_view_loss:
                    total_loss += view_loss
                if hyp.style_view_loss:
                    total_loss += styled_view_loss
                if hyp.cycle_style_view_loss:
                    total_loss += styled_view_loss_cycle

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

