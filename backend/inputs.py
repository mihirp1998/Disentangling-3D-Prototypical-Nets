from backend import readers
# import tensorflow as tf
import numpy as np
np.random.seed(seed=1)
import torch
from torch.utils.data import DataLoader
import hyperparams as hyp
import utils_basic
import pickle
import os
import ipdb
import random
st = ipdb.set_trace
import utils_improc
import utils_geom
from scipy.misc import imresize
import utils_vox

class TFRecordDataset():

    def __init__(self, dataset_path, shuffle=True, val=False):
        with open(dataset_path) as f:
            content = f.readlines()
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        self.nRecords = nRecords
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
            
        dataset = tf.data.TFRecordDataset(
            records,
            compression_type="GZIP"
        ).repeat()
        
        if val:
            num_threads = 1
        else:
            num_threads = 4

        if hyp.dataset_name=='carla' or hyp.dataset_name=='kitti' or hyp.dataset_name=='clevr':
            dataset = dataset.map(readers.carla_parser,
                                  num_parallel_calls=num_threads)
        else:
            assert(False) # reader not ready yet
            
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(hyp.B)
        self.dataset = dataset
        self.iterator = None
        self.sess = tf.Session()

        self.iterator = self.dataset.make_one_shot_iterator()
        self.batch_to_run = self.iterator.get_next()

    def __getitem__(self, index):
        try:
            batch = self.sess.run(self.batch_to_run)
        except tf.errors.OutOfRangeError:
            self.iterator = self.dataset.make_one_shot_iterator()
            self.batch_to_run = self.iterator.get_next()
            batch = self.sess.run(self.batch_to_run)

        batch_torch = []
        for b in batch:
            batch_torch.append(torch.tensor(b))

        d = {}
        [d['pix_T_cams'],
         d['cam_T_velos'],
         d['origin_T_camRs'],
         d['origin_T_camXs'],
         d['rgb_camRs'],
         d['rgb_camXs'],
         d['xyz_veloXs'],
         d['boxes3D'], 
         d['tids'], 
         d['scores'], 
         ] = batch_torch

        if hyp.do_time_flip:
            d = random_time_flip_batch(d)
            
        return d

    def __len__(self):
        return 10000000000 #never end 

class NpzRecordDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle):
        with open(dataset_path) as f:
            content = f.readlines()
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        # st()
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
        # st()
        self.records = records
        self.shuffle = shuffle    

    def __getitem__(self, index):
        if hyp.dataset_name=='kitti'or hyp.dataset_name=='clevr' or  hyp.dataset_name=='real'  or hyp.dataset_name=="bigbird" or hyp.dataset_name=="carla" or hyp.dataset_name =="carla_mix"  or hyp.dataset_name == "carla_det" or hyp.dataset_name =="replica" or hyp.dataset_name =="clevr_vqa":
            # print(index)
            filename = self.records[index]
            d = pickle.load(open(filename,"rb"))
            d = dict(d)
        # elif hyp.dataset_name=="carla":
        #     filename = self.records[index]
        #     d = np.load(filename)
        #     d = dict(d)
                
        #     d['rgb_camXs_raw'] = d['rgb_camXs']
        #     d['pix_T_cams_raw'] = d['pix_T_cams']
        #     d['tree_seq_filename'] = "dummy_tree_filename"
        #     d['origin_T_camXs_raw'] = d['origin_T_camXs']
        #     d['camR_T_origin_raw'] = utils_geom.safe_inverse(torch.from_numpy(d['origin_T_camRs'])).numpy()
        #     d['xyz_camXs_raw'] = d['xyz_camXs']

        else:
            assert(False) # reader not ready yet
        
        # st()
        # if hyp.save_gt_occs:
            # pickle.dump(d,open(filename, "wb"))
            # st()
        # st()
        if hyp.use_gt_occs:
            __p = lambda x: utils_basic.pack_seqdim(x, 1)
            __u = lambda x: utils_basic.unpack_seqdim(x, 1)

            B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
            PH, PW = hyp.PH, hyp.PW
            K = hyp.K
            BOX_SIZE = hyp.BOX_SIZE
            Z, Y, X = hyp.Z, hyp.Y, hyp.X
            Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
            Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
            D = 9
            pix_T_cams = torch.from_numpy(d["pix_T_cams_raw"]).unsqueeze(0).cuda().to(torch.float)
            camRs_T_origin = torch.from_numpy(d["camR_T_origin_raw"]).unsqueeze(0).cuda().to(torch.float)
            origin_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_origin)))
            origin_T_camXs = torch.from_numpy(d["origin_T_camXs_raw"]).unsqueeze(0).cuda().to(torch.float)
            camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
            camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
            camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
            camX0_T_camRs = camXs_T_camRs[:,0]
            camX1_T_camRs = camXs_T_camRs[:,1]
            camR_T_camX0  = utils_geom.safe_inverse(camX0_T_camRs)
            xyz_camXs = torch.from_numpy(d["xyz_camXs_raw"]).unsqueeze(0).cuda().to(torch.float)
            xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
            depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
            dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
            occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
            occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
            occRs_half = torch.max(occRs_half,dim=1).values.squeeze(0)
            occ_complete = occRs_half.cpu().numpy()

            # st()

        if hyp.do_empty:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
                'empty_rgb_camXs_raw',
                'empty_xyz_camXs_raw',            
            ]
        else:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
            ]
        
        # if hyp.do_time_flip:
        #     d = random_time_flip_single(d,item_names)
        # if the sequence length > 2, select S frames
        # filename = d['raw_seq_filename']
        original_filename = filename
        if hyp.dataset_name =="carla_mix"  or hyp.dataset_name == "carla_det":
            bbox_origin_gt = d['bbox_origin']
            if 'bbox_origin_predicted' in d:
                bbox_origin_predicted = d['bbox_origin_predicted']
            else:
                bbox_origin_predicted = []
            classes = d['obj_name']
            
            if isinstance(classes,str):
                classes = [classes]
            # st()

            d['tree_seq_filename'] = "temp"
        if hyp.dataset_name =="replica":
            d['tree_seq_filename'] = "temp"
            object_category = d['object_category_names']
            bbox_origin = d['bbox_origin']

        if hyp.dataset_name =="clevr_vqa":
            d['tree_seq_filename'] = "temp"
            pix_T_cams = d['pix_T_cams_raw']
            num_cams = pix_T_cams.shape[0]
            # padding_1 = torch.zeros([num_cams,1,3])
            # padding_2 = torch.zeros([num_cams,4,1])
            # padding_2[:,3] = 1.0
            # st()
            # pix_T_cams = torch.cat([pix_T_cams,padding_1],dim=1)
            # pix_T_cams = torch.cat([pix_T_cams,padding_2],dim=2)
            # st()
            shape_name = d['shape_list']
            color_name = d['color_list']
            material_name = d['material_list']
            all_name = []
            all_style = []
            for index in range(len(shape_name)):
                name = shape_name[index] + "/" + color_name[index] + "_" + material_name[index]
                style_name  = color_name[index] + "_" + material_name[index]
                all_name.append(name)
                all_style.append(style_name)

            # st()
            
            if hyp.do_shape:
                class_name = shape_name
            elif hyp.do_color:
                class_name = color_name
            elif hyp.do_material:
                class_name = material_name
            elif hyp.do_style:
                class_name = all_style
            else:
                class_name = all_name

            object_category = class_name
            bbox_origin = d['bbox_origin']
            # st()

        if hyp.dataset_name=="carla":
            camR_index = d['camR_index']
            rgb_camtop = d['rgb_camXs_raw'][camR_index:camR_index+1]
            origin_T_camXs_top = d['origin_T_camXs_raw'][camR_index:camR_index+1]
            # predicted_box  = d['bbox_origin_predicted']
            predicted_box = []    
        filename = d['tree_seq_filename']
        if hyp.do_2d_style_munit:
            d,indexes = non_random_select_single(d, item_names, num_samples=hyp.S)
        
        # st()
        if hyp.fixed_view:
            d,indexes = non_random_select_single(d, item_names, num_samples=hyp.S)
        elif self.shuffle or hyp.randomly_select_views:
            d,indexes = random_select_single(d, item_names, num_samples=hyp.S)
        else:
            d,indexes = non_random_select_single(d, item_names, num_samples=hyp.S)

        filename_g = "/".join([original_filename,str(indexes[0])])
        filename_e = "/".join([original_filename,str(indexes[1])])

        rgb_camXs = d['rgb_camXs_raw']
        # move channel dim inward, like pytorch wants
        # rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])

        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = rgb_camXs[:,:3]
        rgb_camXs = utils_improc.preprocess_color(rgb_camXs)

        if hyp.dataset_name=="carla":
            rgb_camtop = np.transpose(rgb_camtop, axes=[0, 3, 1, 2])
            rgb_camtop = rgb_camtop[:,:3]
            rgb_camtop = utils_improc.preprocess_color(rgb_camtop)
            d['rgb_camtop'] = rgb_camtop
            d['origin_T_camXs_top'] = origin_T_camXs_top
            if len(predicted_box) == 0:
                predicted_box = np.zeros([hyp.N,6])
                score = np.zeros([hyp.N]).astype(np.float32)
            else:
                num_boxes = predicted_box.shape[0]
                score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                predicted_box = np.pad(predicted_box,[[0,hyp.N-num_boxes],[0,0]])
            d['predicted_box'] = predicted_box.astype(np.float32)
            d['predicted_scores'] = score.astype(np.float32)
        if hyp.dataset_name == "clevr_vqa":
            num_boxes = bbox_origin.shape[0]
            bbox_origin = np.array(bbox_origin)
            score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
            bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0],[0,0]])
            object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")

            d['gt_box'] = bbox_origin.astype(np.float32)
            d['gt_scores'] = score.astype(np.float32)
            d['classes']  = list(object_category)

        if hyp.dataset_name=="replica":
            if len(bbox_origin) == 0:
                score = np.zeros([hyp.N])
                bbox_origin = np.zeros([hyp.N,6])
                object_category = ["0"]*hyp.N
                object_category = np.array(object_category)
            else:
                num_boxes = len(bbox_origin)
                bbox_origin = torch.stack(bbox_origin).numpy().squeeze(1).squeeze(1).reshape([num_boxes,6])
                bbox_origin = np.array(bbox_origin)
                score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0]])
                object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
            d['gt_box'] = bbox_origin.astype(np.float32)
            d['gt_scores'] = score.astype(np.float32)
            d['classes']  = list(object_category)
            # st()

        if hyp.dataset_name =="carla_mix"  or hyp.dataset_name == "carla_det":
            bbox_origin_predicted = bbox_origin_predicted[:3]
            if len(bbox_origin_gt.shape) ==1:
                bbox_origin_gt = np.expand_dims(bbox_origin_gt,0)
            num_boxes = bbox_origin_gt.shape[0]
            # st()
            score_gt = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
            bbox_origin_gt = np.pad(bbox_origin_gt,[[0,hyp.N-num_boxes],[0,0]])
            # st()
            classes = np.pad(classes,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")

            if len(bbox_origin_predicted) == 0:
                bbox_origin_predicted = np.zeros([hyp.N,6])
                score_pred = np.zeros([hyp.N]).astype(np.float32)
            else:
                num_boxes = bbox_origin_predicted.shape[0]
                score_pred = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                bbox_origin_predicted = np.pad(bbox_origin_predicted,[[0,hyp.N-num_boxes],[0,0]])
                
            d['predicted_box'] = bbox_origin_predicted.astype(np.float32)
            d['predicted_scores'] = score_pred.astype(np.float32)            
            d['gt_box'] = bbox_origin_gt.astype(np.float32)
            d['gt_scores'] = score_gt.astype(np.float32)
            d['classes']  = list(classes)

        d['rgb_camXs_raw'] = rgb_camXs

        if hyp.dataset_name!="carla" and hyp.do_empty:
            empty_rgb_camXs = d['empty_rgb_camXs_raw']
            # move channel dim inward, like pytorch wants
            empty_rgb_camXs = np.transpose(empty_rgb_camXs, axes=[0, 3, 1, 2])
            empty_rgb_camXs = empty_rgb_camXs[:,:3]
            empty_rgb_camXs = utils_improc.preprocess_color(empty_rgb_camXs)
            d['empty_rgb_camXs_raw'] = empty_rgb_camXs
        # st()
        if hyp.use_gt_occs:
            d['occR_complete'] = occ_complete
        d['tree_seq_filename'] = filename
        d['filename_e'] = filename_e
        d['filename_g'] = filename_g
        return d

    def __len__(self):
        return len(self.records)




class NpzRecordDataset_Empty(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle):
        with open(dataset_path) as f:
            content = f.readlines()
        records = [hyp.dataset_location + '/' + line.strip() for line in content]
        # st()
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        nCheck = np.min([nRecords, 1000])
        for record in records[:nCheck]:
            assert os.path.isfile(record), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
        # st()
        import socket
        val = socket.gethostname()
        if "Alien" in val:
            self.empty_scene = '/media/mihir/dataset/clevr_lang/npys/aa/CLEVR_new_000046.p'
            self.empty_scene = '/media/mihir/dataset/clevr_lang/npys/aa/empty_480_a_15908568436475785.p'
        else:
            self.empty_scene = '/projects/katefgroup/datasets/clevr_vqa/raw/npys/empty_480_a/empty_480_a_15908568436475785.p'
        self.records = records
        self.shuffle = shuffle    

    def __getitem__(self, index):
        if hyp.dataset_name=='kitti'or hyp.dataset_name=='clevr' or  hyp.dataset_name=='real'  or hyp.dataset_name=="bigbird" or hyp.dataset_name=="carla" or hyp.dataset_name =="carla_mix" or hyp.dataset_name =="replica" or hyp.dataset_name =="clevr_vqa"  or hyp.dataset_name == "carla_det": 
            # print(index)
            # st()
            filename = self.records[index]
            d = pickle.load(open(filename,"rb"))
            d = dict(d)

            d_empty = pickle.load(open(self.empty_scene,"rb"))
            d_empty = dict(d_empty)
            # st()
        # elif hyp.dataset_name=="carla":
        #     filename = self.records[index]
        #     d = np.load(filename)
        #     d = dict(d)
                
        #     d['rgb_camXs_raw'] = d['rgb_camXs']
        #     d['pix_T_cams_raw'] = d['pix_T_cams']
        #     d['tree_seq_filename'] = "dummy_tree_filename"
        #     d['origin_T_camXs_raw'] = d['origin_T_camXs']
        #     d['camR_T_origin_raw'] = utils_geom.safe_inverse(torch.from_numpy(d['origin_T_camRs'])).numpy()
        #     d['xyz_camXs_raw'] = d['xyz_camXs']

        else:
            assert(False) # reader not ready yet
        

        if hyp.do_empty:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
                'empty_rgb_camXs_raw',
                'empty_xyz_camXs_raw',            
            ]
        else:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
            ]


        if hyp.use_gt_occs:
            __p = lambda x: utils_basic.pack_seqdim(x, 1)
            __u = lambda x: utils_basic.unpack_seqdim(x, 1)

            B, H, W, V, S, N = hyp.B, hyp.H, hyp.W, hyp.V, hyp.S, hyp.N
            PH, PW = hyp.PH, hyp.PW
            K = hyp.K
            BOX_SIZE = hyp.BOX_SIZE
            Z, Y, X = hyp.Z, hyp.Y, hyp.X
            Z2, Y2, X2 = int(Z/2), int(Y/2), int(X/2)
            Z4, Y4, X4 = int(Z/4), int(Y/4), int(X/4)
            D = 9
            pix_T_cams = torch.from_numpy(d["pix_T_cams_raw"]).unsqueeze(0).cuda().to(torch.float)
            camRs_T_origin = torch.from_numpy(d["camR_T_origin_raw"]).unsqueeze(0).cuda().to(torch.float)
            origin_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_origin)))
            origin_T_camXs = torch.from_numpy(d["origin_T_camXs_raw"]).unsqueeze(0).cuda().to(torch.float)
            camX0_T_camXs = utils_geom.get_camM_T_camXs(origin_T_camXs, ind=0)
            camRs_T_camXs = __u(torch.matmul(utils_geom.safe_inverse(__p(origin_T_camRs)), __p(origin_T_camXs)))
            camXs_T_camRs = __u(utils_geom.safe_inverse(__p(camRs_T_camXs)))
            camX0_T_camRs = camXs_T_camRs[:,0]
            camX1_T_camRs = camXs_T_camRs[:,1]
            camR_T_camX0  = utils_geom.safe_inverse(camX0_T_camRs)
            xyz_camXs = torch.from_numpy(d["xyz_camXs_raw"]).unsqueeze(0).cuda().to(torch.float)
            xyz_camRs = __u(utils_geom.apply_4x4(__p(camRs_T_camXs), __p(xyz_camXs)))
            depth_camXs_, valid_camXs_ = utils_geom.create_depth_image(__p(pix_T_cams), __p(xyz_camXs), H, W)
            dense_xyz_camXs_ = utils_geom.depth2pointcloud(depth_camXs_, __p(pix_T_cams))
            occXs = __u(utils_vox.voxelize_xyz(__p(xyz_camXs), Z, Y, X))
            occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))
            occRs_half = torch.max(occRs_half,dim=1).values.squeeze(0)
            occ_complete = occRs_half.cpu().numpy()

        
        # if hyp.do_time_flip:
        #     d = random_time_flip_single(d,item_names)
        # if the sequence length > 2, select S frames
        # filename = d['raw_seq_filename']
        original_filename = filename
        original_filename_empty = self.empty_scene

        # st()
        if hyp.dataset_name =="clevr_vqa":
            d['tree_seq_filename'] = "temp"
            pix_T_cams = d['pix_T_cams_raw']
            num_cams = pix_T_cams.shape[0]
            # padding_1 = torch.zeros([num_cams,1,3])
            # padding_2 = torch.zeros([num_cams,4,1])
            # padding_2[:,3] = 1.0
            # st()
            # pix_T_cams = torch.cat([pix_T_cams,padding_1],dim=1)
            # pix_T_cams = torch.cat([pix_T_cams,padding_2],dim=2)
            # st()
            shape_name = d['shape_list']
            color_name = d['color_list']
            material_name = d['material_list']
            all_name = []
            all_style = []
            for index in range(len(shape_name)):
                name = shape_name[index] + "/" + color_name[index] + "_" + material_name[index]
                style_name  = color_name[index] + "_" + material_name[index]
                all_name.append(name)
                all_style.append(style_name)

            # st()
            
            if hyp.do_shape:
                class_name = shape_name
            elif hyp.do_color:
                class_name = color_name
            elif hyp.do_material:
                class_name = material_name
            elif hyp.do_style:
                class_name = all_style
            else:
                class_name = all_name

            object_category = class_name
            bbox_origin = d['bbox_origin']
            # bbox_origin = torch.cat([bbox_origin],dim=0)
            # object_category = object_category
            bbox_origin_empty = np.zeros_like(bbox_origin)
            object_category_empty = ['0']
        # st()
        if not hyp.dataset_name =="clevr_vqa":
            filename = d['tree_seq_filename']
            filename_empty = d_empty['tree_seq_filename']
        if hyp.fixed_view:
            d,indexes = non_random_select_single(d, item_names, num_samples=hyp.S)
            d_empty,indexes_empty = specific_select_single_empty(d_empty, item_names, d['origin_T_camXs_raw'], num_samples=hyp.S)


        filename_g = "/".join([original_filename,str(indexes[0])])
        filename_e = "/".join([original_filename,str(indexes[1])])

        filename_g_empty = "/".join([original_filename_empty,str(indexes[0])])
        filename_e_empty = "/".join([original_filename_empty,str(indexes[1])])
        
        rgb_camXs = d['rgb_camXs_raw']
        rgb_camXs_empty = d_empty['rgb_camXs_raw']
        # move channel dim inward, like pytorch wants
        # rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = rgb_camXs[:,:3]
        rgb_camXs = utils_improc.preprocess_color(rgb_camXs)

        rgb_camXs_empty = np.transpose(rgb_camXs_empty, axes=[0, 3, 1, 2])
        rgb_camXs_empty = rgb_camXs_empty[:,:3]
        rgb_camXs_empty = utils_improc.preprocess_color(rgb_camXs_empty)

        if hyp.dataset_name == "clevr_vqa":
            num_boxes = bbox_origin.shape[0]
            bbox_origin = np.array(bbox_origin)
            score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
            bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0],[0,0]])
            object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
            object_category_empty = np.pad(object_category_empty,[[0,hyp.N-1]],lambda x,y,z,m: "0")

            # st()
            score_empty = np.zeros_like(score)
            bbox_origin_empty = np.zeros_like(bbox_origin)
            d['gt_box'] = np.stack([bbox_origin.astype(np.float32),bbox_origin_empty])
            d['gt_scores'] = np.stack([score.astype(np.float32),score_empty])
            try:
                d['classes']  =  np.stack([object_category,object_category_empty]).tolist()
            except Exception as e:
                st()

        d['rgb_camXs_raw'] = np.stack([rgb_camXs,rgb_camXs_empty])
        d['pix_T_cams_raw'] = np.stack([d["pix_T_cams_raw"],d_empty["pix_T_cams_raw"]])
        d['origin_T_camXs_raw'] = np.stack([d["origin_T_camXs_raw"],d_empty["origin_T_camXs_raw"]])
        d['camR_T_origin_raw'] = np.stack([d["camR_T_origin_raw"],d_empty["camR_T_origin_raw"]])
        d['xyz_camXs_raw'] = np.stack([d["xyz_camXs_raw"],d_empty["xyz_camXs_raw"]])
        # d['rgb_camXs_raw'] = rgb_camXs
        # d['tree_seq_filename'] = filename
        if not hyp.dataset_name =="clevr_vqa":
            d['tree_seq_filename'] = [filename,"invalid_tree"]
        else:        
            d['tree_seq_filename'] = ["temp"]
        # st()
        d['filename_e'] = ["temp"]
        d['filename_g'] = ["temp"]
        if hyp.use_gt_occs:
            d['occR_complete'] = np.expand_dims(occ_complete,axis=0)
        return d

    def __len__(self):
        return len(self.records)


class NpzRecordDataset_ContentDriven(torch.utils.data.Dataset):
    def __init__(self, dataset_path, shuffle):
        dataset_dict = pickle.load(open(dataset_path,"rb"))
        dataset_dict_updated = {}
        for key,content in dataset_dict.items():
            records = [hyp.dataset_location + '/' + line.strip() for line in content]
            dataset_dict_updated[key] = records
        nRecords = len(records)
        print('found %d records in %s' % (nRecords, dataset_path))
        self.keys = list(dataset_dict_updated.keys())
        self.smallest_len = min([len(i) for i in dataset_dict_updated.values()])
        self.records = dataset_dict_updated
        self.shuffle = shuffle    

    def __getitem__(self, index):
        if hyp.dataset_name=='kitti'or hyp.dataset_name=='clevr' or  hyp.dataset_name=='real'  or hyp.dataset_name=="bigbird" or hyp.dataset_name=="carla" or hyp.dataset_name =="carla_mix" or hyp.dataset_name =="replica" or hyp.dataset_name =="clevr_vqa"  or hyp.dataset_name == "carla_det":
            key_selected = random.choice(self.keys)
            index_selected = random.choice(list(range(len(self.records[key_selected]))))
            filename_1 = self.records[key_selected][index_selected]
            d_1 = pickle.load(open(filename_1,"rb"))
            d_1 = dict(d_1)
            index_selected = random.choice(list(range(len(self.records[key_selected]))))
            filename_2 = self.records[key_selected][index_selected]
            d_2 = pickle.load(open(filename_2,"rb"))
            d_2 = dict(d_2)
        else:
            assert(False) # reader not ready yet

        d = dict()

        
        
        item_names = [
            'pix_T_cams_raw',
            'origin_T_camXs_raw',
            'camR_T_origin_raw',
            'rgb_camXs_raw',
            'xyz_camXs_raw']

        # if hyp.do_time_flip:
        #     d = random_time_flip_single(d,item_names)
        # if the sequence length > 2, select S frames
        # filename = d['raw_seq_filename']
        # original_filename = filename
        # st()
        filename_1 = d_1['tree_seq_filename']
        filename_2 = d_2['tree_seq_filename']
        
        if hyp.fixed_view:
            d_1,indexes = select_single(d_1, item_names, num_samples=hyp.S)
            d_2,indexes = select_single(d_2, item_names, num_samples=hyp.S)
        elif self.shuffle or hyp.randomly_select_views:
            d_1,indexes = random_select_single(d_1, item_names, num_samples=hyp.S)
            d_2,indexes = random_select_single(d_2, item_names, num_samples=hyp.S)
        else:
            d_1,indexes = non_random_select_single(d_1, item_names, num_samples=hyp.S)
            d_2,indexes = non_random_select_single(d_2, item_names, num_samples=hyp.S)
        # filename_g = "/".join([original_filename,str(indexes[0])])
        # filename_e = "/".join([original_filename,str(indexes[1])])
        rgb_camXs_1 = d_1['rgb_camXs_raw']
        rgb_camXs_1 = np.transpose(rgb_camXs_1, axes=[0, 3, 1, 2])
        rgb_camXs_1 = rgb_camXs_1[:,:3]
        rgb_camXs_1 = utils_improc.preprocess_color(rgb_camXs_1)
        
        rgb_camXs_2 = d_2['rgb_camXs_raw']
        rgb_camXs_2 = np.transpose(rgb_camXs_2, axes=[0, 3, 1, 2])
        rgb_camXs_2 = rgb_camXs_2[:,:3]
        rgb_camXs_2 = utils_improc.preprocess_color(rgb_camXs_2)
        rgb_camXs = np.stack([rgb_camXs_1,rgb_camXs_2])
        # st()

        d['rgb_camXs_raw'] = rgb_camXs
        d['pix_T_cams_raw'] = np.stack([d_1["pix_T_cams_raw"],d_2["pix_T_cams_raw"]])
        d['origin_T_camXs_raw'] = np.stack([d_1["origin_T_camXs_raw"],d_2["origin_T_camXs_raw"]])
        d['camR_T_origin_raw'] = np.stack([d_1["camR_T_origin_raw"],d_2["camR_T_origin_raw"]])
        d['xyz_camXs_raw'] = np.stack([d_1["xyz_camXs_raw"],d_2["xyz_camXs_raw"]])
        # st()
        d['tree_seq_filename'] = [filename_1,filename_2]
        # st()
        d['filename_e'] = "temp"
        d['filename_g'] = "temp"
        return d

    def __len__(self):
        return self.smallest_len


def random_select_single(batch,item_names, num_samples=2):
    num_all = len(batch[item_names[0]]) #total number of frames
    
    batch_new = {}
    # select valid candidate
    if 'valid_pairs' in batch:
        valid_pairs = batch['valid_pairs'] #this is ? x 2
        sample_pair = np.random.randint(0, len(valid_pairs), 1).squeeze()
        sample_id = valid_pairs[sample_pair, :] #this is length-2
    else:
        sample_id = range(num_all)

    final_sample = np.random.choice(sample_id, size=num_samples, replace=False)

    if num_samples > len(sample_id):
        print('Inputs.py. Warning: S larger than valid frames number')

    for item_name in item_names:
        item = batch[item_name]
        item = item[final_sample]
        batch_new[item_name] = item

    return batch_new,final_sample

def non_random_select_single(batch, item_names, num_samples=2):
    num_all = len(batch[item_names[0]]) #total number of frames
    
    batch_new = {}
    # select valid candidate
    if 'valid_pairs' in batch:
        valid_pairs = batch['valid_pairs'] #this is ? x 2
        sample_pair = -1
        sample_id = valid_pairs[sample_pair, :] #this is length-2
    else:
        sample_id = range(num_all)

    if len(sample_id) > num_samples:
        final_sample = sample_id[:num_samples]
    else:
        final_sample = sample_id

    if num_samples > len(sample_id):
        print('Inputs.py. Warning: S larger than valid frames number')

    for item_name in item_names:
        item = batch[item_name]
        item = item[final_sample]
        batch_new[item_name] = item

    return batch_new,final_sample

def specific_select_single_empty(batch, item_names, ext, num_samples=2):
    num_all = len(batch[item_names[0]]) #total number of frames
    
    first_index = np.sum(np.sum(ext[0]==batch['origin_T_camXs_raw'],axis=1),axis=1).tolist().index(16)
    second_index = np.sum(np.sum(ext[1]==batch['origin_T_camXs_raw'],axis=1),axis=1).tolist().index(16)

    batch_new = {}
    # st
    # select valid candidate
    if 'valid_pairs' in batch:
        valid_pairs = batch['valid_pairs'] #this is ? x 2
        sample_pair = -1
        sample_id = valid_pairs[sample_pair, :] #this is length-2
    else:
        sample_id = range(num_all)
    final_sample = [first_index,second_index]
    # if len(sample_id) > num_samples:
    #     final_sample = sample_id[-num_samples:]
    # else:
    #     final_sample = sample_id

    if num_samples > len(sample_id):
        print('Inputs.py. Warning: S larger than valid frames number')

    for item_name in item_names:
        item = batch[item_name]
        item = item[final_sample]
        batch_new[item_name] = item

    return batch_new,final_sample


def random_time_flip_batch(batch, item_names):
    # let's do this for the whole batch at once, for simplicity
    # do_flip = tf.cast(tf.random_uniform([1],minval=0,maxval=2,dtype=tf.int32), tf.bool)
    do_flip = torch.rand(1)
    

    for item_name in item_names:
        item = batch[item_name]
        if do_flip > 0.5:
            # flip along the seq dim
            item = item.flip(1)
        batch[item_name] = item
        
    return batch

def random_time_flip_single(batch, item_names):
    # let's do this for the whole batch at once, for simplicity
    # do_flip = tf.cast(tf.random_uniform([1],minval=0,maxval=2,dtype=tf.int32), tf.bool)
    do_flip = torch.rand(1)
    
    for item_name in item_names:
        item = batch[item_name]
        if do_flip > 0.5:
            if torch.is_tensor(item):
                # flip along the seq dim
                item = item.flip(0)
            else: #numpy array
                item = np.flip(item, axis=0)
        batch[item_name] = item
        
    return batch

def specific_select_single(batch,item_names, index):
    num_all = len(batch[item_names[0]]) #total number of frames
    batch_new = {}
    for item_name in item_names:
        item = batch[item_name]
        item = item[index]
        batch_new[item_name] = item
    return batch_new


def merge_e_g(d_e,d_g,item_names):
    d = {}    
    for item_name in item_names:
        d_e_item = d_e[item_name]
        d_g_item = d_g[item_name]
        d_item = np.stack([d_g_item,d_e_item])
        d[item_name] = d_item
    return d

def get_bbox(bbox_origin,object_category):
    if len(bbox_origin) == 0:
        score = np.zeros([hyp.N])
        bbox_origin = np.zeros([hyp.N,6])
        object_category = ["0"]*hyp.N
        # st()
        object_category = np.array(object_category)
    else:
        num_boxes = len(bbox_origin)
        # st()
        bbox_origin = torch.stack(bbox_origin).numpy().squeeze(1).squeeze(1).reshape([num_boxes,6])
        bbox_origin = np.array(bbox_origin)
        score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
        bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0]])
        object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
    return bbox_origin,score,object_category

def get_bbox2(bbox_origin,object_category):
    if len(bbox_origin) == 0:
        score = np.zeros([hyp.N])
        bbox_origin = np.zeros([hyp.N,6])
        object_category = ["0"]*hyp.N
        # st()
        object_category = np.array(object_category)
    else:
        num_boxes = bbox_origin.shape[0]
        bbox_origin = np.array(bbox_origin)
        score = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
        bbox_origin = np.pad(bbox_origin,[[0,hyp.N-num_boxes],[0,0],[0,0]])
        object_category = np.pad(object_category,[[0,hyp.N-num_boxes]],lambda x,y,z,m: "0")
    return bbox_origin,score,object_category


class NpzCustomRecordDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, shuffle):
        nRecords = len(filenames)
        print('found %d records' % (nRecords))
        nCheck = np.min([nRecords, 1000])
        if hyp.emb_moc.own_data_loader:
            for record in filenames[:nCheck]:
                assert os.path.isfile(record[0]), 'Record at %s was not found' % record
                assert os.path.isfile(record[1]), 'Record at %s was not found' % record
        else:
            for record in filenames[:nCheck]:
                assert os.path.isfile("/".join(record[0].split("/")[:-1])), 'Record at %s was not found' % record
                assert os.path.isfile("/".join(record[1].split("/")[:-1])), 'Record at %s was not found' % record
        print('checked the first %d, and they seem to be real files' % (nCheck))
        self.records = filenames
        self.shuffle = shuffle

    def __getitem__(self, index):
        if hyp.dataset_name=='carla' or hyp.dataset_name=='kitti'or hyp.dataset_name=='clevr' or hyp.dataset_name=='real' or  hyp.dataset_name=="bigbird" or  hyp.dataset_name=="carla_mix"  or  hyp.dataset_name=="replica" or  hyp.dataset_name=="clevr_vqa"  or hyp.dataset_name == "carla_det":
            filename = self.records[index]
            filename_e,filename_g = filename
            if hyp.emb_moc.own_data_loader:
                d_e = pickle.load(open(filename_e,"rb"))
                d_g = pickle.load(open(filename_g,"rb"))
                index_e_parts = str(np.random.randint(0,hyp.NUM_VIEWS))
                index_g_parts = str(np.random.randint(0,hyp.NUM_VIEWS))
            else:
                filename_e_parts = filename_e.split("/")
                index_e_parts = filename_e_parts[-1]
                main_filename_e =  "/".join(filename_e_parts[:-1])

                filename_g_parts = filename_g.split("/")
                index_g_parts = filename_g_parts[-1]
                main_filename_g =  "/".join(filename_g_parts[:-1])

                d_e = pickle.load(open(main_filename_e,"rb"))
                d_g = pickle.load(open(main_filename_g,"rb"))
        else:
            assert(False) # reader not ready yet
        if hyp.do_empty:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
                'empty_rgb_camXs_raw',
                'empty_xyz_camXs_raw',            
            ]
        else:
            item_names = [
                'pix_T_cams_raw',
                'origin_T_camXs_raw',
                'camR_T_origin_raw',
                'rgb_camXs_raw',
                'xyz_camXs_raw',
            ]

        d_e = dict(d_e)
        d_g = dict(d_g)
        
        if not hyp.dataset_name == "carla_mix" and not hyp.dataset_name == "replica" and not hyp.dataset_name == "clevr_vqa" and not hyp.dataset_name == "carla_det":
            filename_de = d_e['tree_seq_filename']
            filename_dg = d_g['tree_seq_filename']
        else:
            filename_de = filename_e
            filename_dg = filename_g
        
        if hyp.dataset_name =="carla_mix"  or hyp.dataset_name == "carla_det":
            bbox_origin_gt = d_e['bbox_origin']
            bbox_origin_predicted = d_e['bbox_origin_predicted']
        
        if hyp.dataset_name =="replica":
            obj_cat_name_e = d_e['object_category_names']
            obj_cat_name_g = d_g['object_category_names']
            bbox_origin_gt_e = d_e['bbox_origin']
            bbox_origin_gt_g = d_g['bbox_origin']
        
        if hyp.dataset_name =="clevr_vqa":
            obj_cat_name_e = d_e['shape_list']
            obj_cat_name_g = d_g['shape_list']
            bbox_origin_gt_e = d_e['bbox_origin']
            bbox_origin_gt_g = d_g['bbox_origin']            

            # d['tree_seq_filename'] = "temp"

        d_e = specific_select_single(d_e, item_names, int(index_e_parts))
        d_g = specific_select_single(d_g, item_names, int(index_g_parts))

        d = merge_e_g(d_e,d_g,item_names)
        
        # merge is g and e
        if hyp.dataset_name =="carla_mix"  or hyp.dataset_name == "carla_det":
            bbox_origin_predicted = bbox_origin_predicted[:3]
            if len(bbox_origin_gt.shape) ==1:
                bbox_origin_gt = np.expand_dims(bbox_origin_gt,0)
            num_boxes = bbox_origin_gt.shape[0]
            # st()
            score_gt = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
            bbox_origin_gt = np.pad(bbox_origin_gt,[[0,hyp.N-num_boxes],[0,0]])

            if len(bbox_origin_predicted) == 0:
                bbox_origin_predicted = np.zeros([hyp.N,6])
                score_pred = np.zeros([hyp.N]).astype(np.float32)
            else:
                num_boxes = bbox_origin_predicted.shape[0]
                score_pred = np.pad(np.ones([num_boxes]),[0,hyp.N-num_boxes])
                bbox_origin_predicted = np.pad(bbox_origin_predicted,[[0,hyp.N-num_boxes],[0,0]])
            d['predicted_box'] = bbox_origin_predicted.astype(np.float32)
            d['predicted_scores'] = score_pred.astype(np.float32)            
            d['gt_box'] = bbox_origin_gt.astype(np.float32)
            d['gt_scores'] = score_gt.astype(np.float32)


        d["filename_e"] = filename_de
        d["filename_g"] = filename_dg
        d["tree_seq_filename"] = filename_dg
        rgb_camXs = d['rgb_camXs_raw']

        if hyp.dataset_name =="replica":
            bbox_origin_e, score_e, object_category_e = get_bbox(bbox_origin_gt_e,obj_cat_name_e)
            bbox_origin_g, score_g, object_category_g = get_bbox(bbox_origin_gt_g,obj_cat_name_g)
            d['object_category_names_e'] = list(object_category_e)
            d['object_category_names_g'] = list(object_category_g)
            d['bbox_origin_e'] = bbox_origin_e
            d['bbox_origin_g'] = bbox_origin_g
            d['scores_e'] = score_e
            d['scores_g'] = score_g

        if  hyp.dataset_name =="clevr_vqa":
            bbox_origin_e, score_e, object_category_e = get_bbox2(bbox_origin_gt_e,obj_cat_name_e)
            bbox_origin_g, score_g, object_category_g = get_bbox2(bbox_origin_gt_g,obj_cat_name_g)
            d['object_category_names_e'] = list(object_category_e)
            d['object_category_names_g'] = list(object_category_g)
            d['bbox_origin_e'] = bbox_origin_e
            d['bbox_origin_g'] = bbox_origin_g
            d['scores_e'] = score_e
            d['scores_g'] = score_g


        # move channel dim inward, like pytorch wants
        # rgb_camRs = np.transpose(rgb_camRs, axes=[0, 3, 1, 2])
        rgb_camXs = np.transpose(rgb_camXs, axes=[0, 3, 1, 2])
        rgb_camXs = rgb_camXs[:,:3]
        # rgb_camRs = utils_improc.preprocess_color(rgb_camRs)
        rgb_camXs = utils_improc.preprocess_color(rgb_camXs)

        d['rgb_camXs_raw'] = rgb_camXs

        d['index_val'] = index

        if hyp.do_empty:
            empty_rgb_camXs = d['empty_rgb_camXs_raw']
            empty_rgb_camXs = np.transpose(empty_rgb_camXs, axes=[0, 3, 1, 2])
            empty_rgb_camXs = empty_rgb_camXs[:,:3]
            empty_rgb_camXs = utils_improc.preprocess_color(empty_rgb_camXs)
            d['empty_rgb_camXs_raw'] = empty_rgb_camXs
        return d

    def __len__(self):
        return len(self.records)

def get_inputs():
    dataset_format = hyp.dataset_format
    all_set_inputs = {}
    for set_name in hyp.set_names:
        if hyp.sets_to_run[set_name]:
            data_path = hyp.data_paths[set_name]
            shuffle = hyp.shuffles[set_name]
            if dataset_format == 'tf':
                all_set_inputs[set_name] = TFRecordDataset(dataset_path=data_path, shuffle=shuffle)
            elif dataset_format == 'npz':
                if hyp.do_debug:
                    if hyp.do_match_det:
                        assert hyp.B ==1 
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset_Empty(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)                    
                    elif hyp.debug_match:
                        assert hyp.B ==1 
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset_Empty(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)
                    elif hyp.typeVal == "content":
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset_ContentDriven(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)
                    else:                
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)
                else:
                    if hyp.do_match_det:
                        assert hyp.B ==1 
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset_Empty(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)                        
                    elif hyp.debug_match:
                        assert hyp.B ==1 
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset_Empty(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)
                    elif hyp.typeVal == "content":
                        all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset_ContentDriven(dataset_path=data_path, shuffle=shuffle), \
                        shuffle=shuffle, batch_size=hyp.B, num_workers=1, pin_memory=True, drop_last=True)
                    else:
                        if hyp.remove_air:
                            all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                            shuffle=shuffle, batch_size=hyp.B, num_workers=0, pin_memory=True, drop_last=True)                            
                        else:
                            all_set_inputs[set_name] = torch.utils.data.DataLoader(dataset=NpzRecordDataset(dataset_path=data_path, shuffle=shuffle), \
                            shuffle=shuffle, batch_size=hyp.B, num_workers=1, pin_memory=True, drop_last=True)
            else:
                assert False #what is the data format?
    return all_set_inputs


def get_custom_inputs(filenames):
    if hyp.do_debug:
        all_set_inputs = torch.utils.data.DataLoader(dataset=NpzCustomRecordDataset(filenames=filenames, shuffle=hyp.max.shuffle), \
        shuffle=hyp.max.shuffle, batch_size=hyp.max.B, num_workers=0, pin_memory=True, drop_last=True)
    else:
        all_set_inputs = torch.utils.data.DataLoader(dataset=NpzCustomRecordDataset(filenames=filenames, shuffle=hyp.max.shuffle), \
        shuffle=hyp.max.shuffle, batch_size=hyp.max.B, num_workers=1, pin_memory=True, drop_last=True)
    return all_set_inputs