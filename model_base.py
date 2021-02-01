import time
import numpy as np
import hyperparams as hyp
import torch
from tensorboardX import SummaryWriter
from backend import saverloader, inputs
from backend import inputs as load_inputs
# from backend.double_pool import DoublePool
from torchvision import datasets, transforms
from DoublePool import DoublePool_O
from DoublePool import MOC_DICT,MOC_QUEUE_NORMAL
from DoublePool import ClusterPool
from DoublePool import DetPool
import utils_basic
import socket
import time
import torch.nn.functional as F
import pickle
import utils_eval
import utils_improc
import utils_basic
import ipdb
st = ipdb.set_trace
from collections import defaultdict
import cross_corr
np.set_printoptions(precision=2)
EPS = 1e-6
np.random.seed(0)
MAX_QUEUE = 10 # how many items before the summaryWriter flushes


class Model(object):
    def __init__(self, checkpoint_dir, log_dir):
        print('------ CREATING NEW MODEL ------')
        print(hyp.name)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.all_inputs = inputs.get_inputs()
        print("------ Done getting inputs ------")
        if hyp.moc:
            self.poolvox_moc = MOC_QUEUE_NORMAL(hyp.moc_qsize)
        if hyp.moc_2d:
            self.poolvox_moc_2d = MOC_QUEUE_NORMAL(hyp.moc_qsize)

        if hyp.offline_cluster:
            self.cluster_pool = ClusterPool(hyp.offline_cluster_pool_size)
        if hyp.offline_cluster_eval:
            self.info_dict = defaultdict(lambda:[])
        if hyp.max.hardmining  or hyp.hard_eval or hyp.hard_vis:
            self.mbr = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE-2*hyp.max.margin, hyp.BOX_SIZE-2*hyp.max.margin, hyp.BOX_SIZE-2*hyp.max.margin)
            self.mbr16 = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE, hyp.BOX_SIZE, hyp.BOX_SIZE)
            self.mbr_unpr = cross_corr.meshgrid_based_rotation(32,32,32)
            self.hpm = hardPositiveMiner.HardPositiveMiner(self.mbr,self.mbr16,self.mbr_unpr)
        elif hyp.do_orientation:
            self.mbr16 = cross_corr.meshgrid_based_rotation(hyp.BOX_SIZE, hyp.BOX_SIZE, hyp.BOX_SIZE)
            self.mbr_unpr = cross_corr.meshgrid_based_rotation(32,32,32)
            self.hpm = None
        else:
            self.mbr16 = None
            self.hpm = None

        if hyp.self_improve_iterate:
            self.pool_det = DetPool(hyp.det_pool_size)

        if hyp.do_eval_recall:
            self.eval_dicts = {}
            if hyp.dataset_name=="bigbird":
                self.recalls = [3, 5, 10]                
            else:
                self.recalls = [10, 20, 30]
            if hyp.do_debug or hyp.low_dict_size:
                self.pool_size = 11
            else:
                self.pool_size = hyp.pool_size
            self.eval_dicts["eval"] = {}
            F = 3
            if hyp.eval_recall_o:
                self.eval_dicts["eval"]['pool3D_e'] = DoublePool_O(self.pool_size)
                self.eval_dicts["eval"]['pool3D_g'] = DoublePool_O(self.pool_size)
            else:
                self.eval_dicts["eval"]['pool3D_e'] = DoublePool(self.pool_size)
                self.eval_dicts["eval"]['pool3D_g'] = DoublePool(self.pool_size)
            self.eval_dicts["eval"]['precision3D'] = np.nan*np.array([0.0, 0.0, 0.0], np.float32)
            self.eval_dicts["eval"]['neighbors3D'] = np.zeros((F*10, F*11, 3), np.float32)
        self.device = torch.device("cuda")

    def init_model_k(self, model_q, model_k):
        param_q = model_q.state_dict()
        model_k.load_state_dict(param_q)

    def infer(self):
        pass
    def momentum_update(self,model_q, model_k, beta = 0.999):
        param_k = model_k.state_dict()
        param_q = model_q.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        model_k.load_state_dict(param_k)

    def ml_loss(self,emb_e,emb_g_key,pool):
        vox_emb, vox_emb_key, classes_key = utils_eval.subsample_embs_voxs_positive(emb_e,emb_g_key, classes= None)

        vox_emb_key_og = vox_emb_key

        vox_emb_key = vox_emb_key.permute(0,2,1)
        vox_emb = vox_emb.permute(0,2,1)

        B,_,_ = vox_emb.shape

        emb_q = vox_emb.reshape(-1,hyp.feat_dim)
        emb_k = vox_emb_key.reshape(-1,hyp.feat_dim)

        N = emb_q.shape[0]

        emb_k = F.normalize(emb_k,dim=1)
        emb_q = F.normalize(emb_q,dim=1)

        l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))

        queue_neg = torch.stack(pool.fetch())

        K = queue_neg.shape[0]

        queue_neg = F.normalize(queue_neg,dim=1)

        l_neg = torch.mm(emb_q, queue_neg.T)
        l_pos = l_pos.view(N, 1)
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to(self.device)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        temp = 0.07
        emb_loss = cross_entropy_loss(logits/temp, labels)
        return emb_loss


    def go(self):
        self.start_time = time.time()
        self.infer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay)
        print("------ Done creating models ------")
        # st()
        self.start_iter = saverloader.load_weights(self.model, self.optimizer)
        if hyp.self_improve_once or hyp.filter_boxes:
          self.model.detnet_target.load_state_dict(self.model.detnet.state_dict())

        if hyp.moc:
            self.init_model_k(self.model,self.model_key)
        
        if hyp.sets_to_run["test"]:
            self.start_iter = 0 
        print("------ Done loading weights ------")
        if hyp.self_improve_iterate:
            exp_set_name = "expectation"
            exp_writer = SummaryWriter(self.log_dir + f'/{exp_set_name}', max_queue=MAX_QUEUE, flush_secs=60)                        
            max_set_name = "maximization_det"
            max_writer = SummaryWriter(self.log_dir + f'/{max_set_name}', max_queue=MAX_QUEUE, flush_secs=60)
            
            self.eval_steps = 0
            self.total_exp_iters = 0 
            hyp.max.B = hyp.B

            inputs = self.all_inputs['train']
            exp_log_freq = hyp.exp_log_freq
            exp_loader = iter(inputs)

            self.max_steps = 0
            self.total_max_iters = 0
            while True:
                hyp.exp_do = True
                hyp.exp_done = False                
                hyp.max_do = False  
                if hyp.exp_do:
                    start_time = time.time()
                    print("EVAL MODE: ")
                    self.eval_steps += 1
                    for step in range(hyp.exp_max_iters):
                        if step % len(inputs) == 0:
                            exp_loader = iter(inputs)                        
                        # st()
                        self.total_exp_iters += 1
                        iter_start_time = time.time()
                        log_this = np.mod(self.eval_steps,exp_log_freq) == 0
                        total_time, read_time, iter_time = 0.0, 0.0, 0.0
                        read_start_time = time.time()
                        
                        try:
                            feed = next(exp_loader)
                        except StopIteration:
                            print("FUCKING ERROR")
                            exp_loader = iter(inputs)
                            feed = next(exp_loader)

                        feed_cuda = {}

                        tree_seq_filename = feed.pop('tree_seq_filename')
                        filename_e = feed.pop('filename_e')
                        filename_g = feed.pop('filename_g')
                
                        for k in feed:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                        read_time = time.time() - read_start_time
                        feed_cuda['tree_seq_filename'] = tree_seq_filename
                        feed_cuda['filename_e'] = filename_e
                        feed_cuda['filename_g'] = filename_g
                        feed_cuda['writer'] = exp_writer
                        feed_cuda['global_step'] = self.total_exp_iters
                        feed_cuda['log_freq'] = exp_log_freq
                        feed_cuda['set_name'] = exp_set_name

                        self.model.eval()
                        with torch.no_grad():
                            loss, results = self.model(feed_cuda)


                        loss_vis = loss.cpu().item()
                        summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                               global_step=feed_cuda['global_step'],
                                               set_name=feed_cuda['set_name'],
                                               log_freq=feed_cuda['log_freq'],
                                               fps=8)                        
                        if results['filenames_g'] is not None:
                            filenames_g = results['filenames_g']
                            filenames_e = results['filenames_e']
                            feat_masks =  results['featR_masks']
                            boxes =  results['filtered_boxes']
                            gt_boxes = results["gt_boxes"]
                            scores =  results['scores']
                            gt_scores = results["gt_scores"]
                            filenames = np.stack([filenames_g,filenames_e],axis=1)
                            self.pool_det.update(feat_masks, boxes, gt_boxes, scores, gt_scores, filenames)

                        
                        print("Expectation: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); pool_fill: [%4d/%4d] Global_Steps: %4d"%(
                                                                                            hyp.name,
                                                                                            step,
                                                                                            hyp.exp_max_iters,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            read_time,
                                                                                            self.pool_det.num,
                                                                                            self.pool_det.pool_size,
                                                                                            self.eval_steps))
                if self.pool_det.is_full():
                    hyp.exp_do = False
                    hyp.exp_done = True                
                    hyp.max_do = False
                else:
                    hyp.exp_do = True
                    hyp.exp_done = False                
                    hyp.max_do = False
                if hyp.exp_done:
                    filenames, feat_masks, boxes, gt_boxes, scores,gt_scores = self.pool_det.fetch()
                    final_filenames = np.stack(filenames)
    
                    hyp.exp_do = False
                    hyp.exp_done = False
                    hyp.max_do = True

                if hyp.max_do:
                    max_loader = load_inputs.get_custom_inputs(final_filenames)
                    max_loader_iter = iter(max_loader)
                    max_log_freq = hyp.maxm_log_freq
                    self.model.train()
                    self.max_steps += 1
                    print("MAX MODE: ")
                    boxes = torch.stack(boxes)
                    scores = torch.stack(scores)
                    feat_masks = torch.stack(feat_masks)
                    gt_boxes = torch.stack(gt_boxes)
                    gt_scores = torch.stack(gt_scores)
                    for step in range(hyp.maxm_max_iters):
                        self.total_max_iters += 1
                        iter_start_time = time.time()
                        try:
                            feed = next(max_loader_iter)
                        except StopIteration:
                            print("FUCKING ERROR")
                            max_loader_iter = iter(max_loader)
                            feed = next(max_loader_iter)

                        feed_cuda = {}                        
                        filename_e = feed.pop('filename_e')
                        filename_g = feed.pop('filename_g')
                        tree_seq_filename = feed.pop('tree_seq_filename')
                        index_val = feed.pop('index_val')

                        for k in feed:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                        feed_cuda['filename_e'] = filename_e
                        feed_cuda['filename_g'] = filename_g
                        feed_cuda['tree_seq_filename'] = tree_seq_filename
                        feed_cuda['writer'] = max_writer
                        feed_cuda['global_step'] = self.total_max_iters
                        feed_cuda['log_freq'] = max_log_freq
                        feed_cuda['set_name'] = max_set_name
                        
                        
                        feed_cuda["sudo_gt_boxes"] = boxes[index_val]
                        feed_cuda["sudo_gt_scores"] = scores[index_val]
                        feed_cuda["feat_mask"] = feat_masks[index_val]

                        feed_cuda["gt_boxes"] = gt_boxes[index_val]
                        feed_cuda["gt_scores"] = gt_scores[index_val]

                        iter_start_time = time.time()

                        final_filenames[index_val[0]][0].split("/")[-2][:-2] == filename_e[0].split("/")[-1][:-5]
                        loss, results = self.model(feed_cuda)
                        # st()
                        total_loss = loss
                        backprop_start_time = time.time()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        backprop_time = time.time()- backprop_start_time
                        iter_time = time.time()- iter_start_time
                        total_time = time.time()-start_time
                        print("Predicted Maximization: %s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Global_Steps: %4d"% (hyp.name,
                                                                                            step,
                                                                                            hyp.maxm_max_iters,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            backprop_time,
                                                                                            total_loss,self.max_steps))
                
                saverloader.save(self.model, self.checkpoint_dir, self.total_max_iters, self.optimizer)

            for writer in set_writers: 
                writer.close()


        else:
            set_nums = []
            set_names = []
            set_inputs = []
            set_writers = []
            set_log_freqs = []
            set_do_backprops = []
            set_dicts = []
            set_loaders = []
            set_fakes = []



            for set_name in hyp.set_names:
                if hyp.sets_to_run[set_name]:
                    set_nums.append(hyp.set_nums[set_name])
                    set_names.append(set_name)
                    set_inputs.append(self.all_inputs[set_name])
                    set_writers.append(SummaryWriter(self.log_dir + '/' + set_name, max_queue=MAX_QUEUE, flush_secs=60))
                    set_log_freqs.append(hyp.log_freqs[set_name])
                    set_do_backprops.append(hyp.sets_to_backprop[set_name])
                    set_dicts.append({})
                    set_loaders.append(iter(set_inputs[-1]))
                    set_fakes.append(False)

            # if hyp.halucinate_vals != 1:
            #     if 'val' == set_names[1]:
            #         for i in range(hyp.halucinate_vals):
            #             set_nums.append(set_nums[1] + i +1)
            #             set_names.append(set_names[1])
            #             set_inputs.append(set_inputs[1])
            #             set_writers.append(set_writers[1])
            #             set_log_freqs.append(set_log_freqs[1])
            #             set_do_backprops.append(set_do_backprops[1])
            #             set_dicts.append(set_dicts[1])
            #             set_loaders.append(set_loaders[1])
            #             set_fakes.append(True)

                # st()
            if hyp.moc and hyp.sets_to_run['train']:
                self.total_embmoc_iters = 0
                embmoc_set_name = "train"
                embmoc_writer = SummaryWriter(self.log_dir + f'/{embmoc_set_name}', max_queue=MAX_QUEUE, flush_secs=60)
                embmoc_input = self.all_inputs[embmoc_set_name]
                embmoc_loader = iter(embmoc_input)
                step =  0
                start_time = time.time()
                while True:
                    step += 1
                    if step % len(embmoc_input) == 0:
                        embmoc_loader = iter(embmoc_input)
                    self.total_embmoc_iters += 1
                    iter_start_time = time.time()
                    total_time, read_time, iter_time = 0.0, 0.0, 0.0
                    read_start_time = time.time()
                    try:
                        feed = next(embmoc_loader)
                    except StopIteration:                        
                        embmoc_loader = iter(embmoc_input)
                        feed = next(embmoc_loader)

                    read_time = time.time() - read_start_time
                    feed_cuda = {}
                    tree_seq_filename = feed.pop('tree_seq_filename')
                    filename_e = feed.pop('filename_e')
                    filename_g = feed.pop('filename_g')
                    if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix"  or hyp.dataset_name == "carla_det":
                        classes = feed.pop('classes')                    
                    for k in feed:
                        feed_cuda[k] = feed[k].cuda(non_blocking=True).float()
                    read_time = time.time() - read_start_time
                    feed_cuda['tree_seq_filename'] = tree_seq_filename
                    feed_cuda['filename_e'] = filename_e
                    feed_cuda['filename_g'] = filename_g
                    feed_cuda['writer'] = embmoc_writer
                    feed_cuda['global_step'] = self.total_embmoc_iters
                    feed_cuda['log_freq'] = 50
                    feed_cuda['set_name'] = "queue_init"
                    
                    if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix"  or hyp.dataset_name == "carla_det":
                        classes = np.transpose(np.array(classes))
                        feed_cuda['classes'] = classes                    

                    self.model_key.eval()
                    with torch.no_grad():                    
                        loss, results_key = self.model_key(feed_cuda)                
                    classes_key = results_key['classes']
                    emb3D_e_key = results_key['emb3D_e'].detach()
                    emb3D_g_key = results_key['emb3D_g'].detach()
                    
                    if hyp.moc_2d:
                        emb2D_e_key = results_key['emb2D_e'].detach()
                        emb2D_g_key = results_key['emb2D_g'].detach()
                    
                    vox_emb_key, classes_key = utils_eval.subsample_embs_voxs(emb3D_e_key, emb3D_g_key, classes= classes_key)
                    
                    if hyp.moc_2d:                
                        vox_emb_key_2d, classes_key_2d = utils_eval.subsample_embs_voxs(emb2D_e_key, emb2D_g_key, classes= classes_key)

                    total_time = time.time() - start_time
                    iter_time = time.time()- iter_start_time
                    
                    self.poolvox_moc.update(vox_emb_key,classes_key)
                    
                    if hyp.moc_2d:
                        self.poolvox_moc_2d.update(vox_emb_key_2d,classes_key_2d)

                    queue_size = self.poolvox_moc.num
                    print("Queue Initialization: %s; Steps: %4d; ttime: %.0f (%.2f, %.2f); Keys: %4d"%(
                                                                                            hyp.name,
                                                                                            step,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            read_time,queue_size))
                    if self.poolvox_moc.is_full():
                        break    

            if hyp.aug_det:
                self.total_aug_iters = 0
                # st()
                list_of_keys = list(self.all_inputs.keys())
                
                if "test" in list_of_keys:
                    aug_set_name = "test"
                elif "train" in list_of_keys:
                    aug_set_name = "train"
                else:
                    aug_set_name = "val"
                
                # st()
                # aug_set_name = "train"
                aug_writer = SummaryWriter(self.log_dir + f'/{aug_set_name}', max_queue=MAX_QUEUE, flush_secs=60)
                aug_input = self.all_inputs[aug_set_name]
                aug_loader = iter(aug_input)
                step =  0
                start_time = time.time()

                hyp.store_obj = True
                while True:
                    step += 1
                    if step % (len(aug_input)+1) == 0:
                        if hyp.fixed_view:
                            hyp.store_obj = False
                            break
                        aug_loader = iter(aug_input)

                    if hyp.aug_object_ent_dis:
                        list_val = self.model.list_aug
                    elif hyp.aug_object_ent:
                        list_val = self.model.list_aug
                    elif hyp.aug_object_dis:
                        list_val = self.model.list_aug_content

                    if hyp.debug_aug:
                        if step % 20 == 0:
                            hyp.store_obj = False
                            break
                    # else:
                    #     if len(list_val) == 500:
                    #         hyp.store_obj = False
                    #         break

                    self.total_aug_iters += 1
                    iter_start_time = time.time()
                    total_time, read_time, iter_time = 0.0, 0.0, 0.0
                    read_start_time = time.time()
                    if hyp.debug_match or hyp.do_match_det:
                        hyp.B = 1
                    feed = next(aug_loader)
                    # st()

                    # st()
                    read_time = time.time() - read_start_time
                    feed_cuda = {}
                    tree_seq_filename = feed.pop('tree_seq_filename')
                    filename_e = feed.pop('filename_e')
                    filename_g = feed.pop('filename_g')
                    if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix" or hyp.dataset_name == "clevr_vqa"  or hyp.dataset_name == "carla_det":
                        if hyp.debug_match or hyp.do_match_det:
                            feed['classes'] = np.array(feed['classes']).reshape([hyp.B*2,hyp.N])
                        classes = feed.pop('classes')
                    
                    if hyp.do_clevr_sta:
                        if hyp.debug_match or hyp.do_match_det:
                            feed_cuda['tree_seq_filename'] = np.array(tree_seq_filename).squeeze(1)
                        else:
                            feed_cuda['tree_seq_filename'] = tree_seq_filename
                    
                    for k in feed:
                        if hyp.debug_match or hyp.do_match_det:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float().squeeze(0)
                        else:
                            feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                    if hyp.debug_match or hyp.do_match_det:
                        hyp.B = hyp.B*2

                    if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix" or hyp.dataset_name == "clevr_vqa"  or hyp.dataset_name == "carla_det":
                        if not hyp.debug_match and  not  hyp.do_match_det:
                            classes = np.transpose(np.array(classes))
                        feed_cuda['classes'] = classes
                        feed_cuda['filename_e'] = filename_e
                        feed_cuda['filename_g'] = filename_g
                    # st()
                    read_time = time.time() - read_start_time
                    feed_cuda['filename_e'] = filename_e
                    feed_cuda['filename_g'] = filename_g
                    feed_cuda['writer'] = aug_writer
                    feed_cuda['global_step'] = self.total_aug_iters
                    feed_cuda['log_freq'] = 50
                    feed_cuda['set_name'] = aug_set_name
                    self.model.eval()
                    with torch.no_grad():
                        loss, results_key = self.model(feed_cuda)
                    # st()
                    if hyp.aug_object_ent_dis:
                        emb3D_Rs = results_key['aug_objects']
                        classes = results_key['classes']

                        total_time = time.time() - start_time
                        iter_time = time.time()- iter_start_time
                        # st()
                        for ind,object_val in enumerate(emb3D_Rs):
                            self.model.list_aug.append(object_val.detach())
                            self.model.list_aug_classes.append(classes[ind])                        
                        emb3D_Rs_content = results_key['aug_objects_content']
                        content_classes = results_key['content_classes']

                        emb3D_Rs_style = results_key['aug_objects_style']
                        style_classes = results_key['style_classes']

                        obj_shapes = results_key['obj_shapes']

                        for i in range(len(content_classes)):
                            try:
                                self.model.list_aug_shapes[content_classes[i]].append(obj_shapes[i])
                                self.model.list_aug_content.append(emb3D_Rs_content[i].detach())
                                self.model.list_aug_style.append(emb3D_Rs_style[i].detach())

                                self.model.list_aug_classes_content.append(content_classes[i])
                                self.model.list_aug_classes_style.append(style_classes[i])
                                if hyp.debug_match  or hyp.do_match_det:
                                    self.model.dict_aug[style_classes[i]] = emb3D_Rs_style[i].detach()
                                    self.model.dict_aug[content_classes[i]] = emb3D_Rs_content[i].detach()
                            except Exception as e:
                                st()                        
                    elif hyp.aug_object_ent:
                        emb3D_Rs = results_key['aug_objects']
                        classes = results_key['classes']

                        total_time = time.time() - start_time
                        iter_time = time.time()- iter_start_time

                        for ind,object_val in enumerate(emb3D_Rs):
                            self.model.list_aug.append(object_val.detach())
                            self.model.list_aug_classes.append(classes[ind])
                    elif hyp.aug_object_dis:
                        emb3D_Rs_content = results_key['aug_objects_content']
                        content_classes = results_key['content_classes']

                        emb3D_Rs_style = results_key['aug_objects_style']
                        style_classes = results_key['style_classes']

                        obj_shapes = results_key['obj_shapes']

                        for i in range(len(content_classes)):
                            try:
                                self.model.list_aug_shapes[content_classes[i]].append(obj_shapes[i])
                                self.model.list_aug_content.append(emb3D_Rs_content[i].detach())
                                self.model.list_aug_style.append(emb3D_Rs_style[i].detach())

                                self.model.list_aug_classes_content.append(content_classes[i])
                                self.model.list_aug_classes_style.append(style_classes[i])
                                if hyp.debug_match  or hyp.do_match_det:
                                    self.model.dict_aug[style_classes[i]] = emb3D_Rs_style[i].detach()
                                    self.model.dict_aug[content_classes[i]] = emb3D_Rs_content[i].detach()
                            except Exception as e:
                                st()

                        total_time = time.time() - start_time
                        iter_time = time.time()- iter_start_time
                    print("List Initialization: %s; Steps: %4d; ttime: %.0f (%.2f, %.2f); Keys: %4d"%(
                                                                                            hyp.name,
                                                                                            step,
                                                                                            total_time,
                                                                                            iter_time,
                                                                                            read_time,len(list_val)))

  
            
            # st()
            for step in range(self.start_iter+1, hyp.max_iters+1):
                for i, (set_input) in enumerate(set_inputs):
                    if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                        if hyp.add_det_boxes or hyp.store_content_style_range or hyp.save_gt_occs:
                            st()
                            print("entire iteration over dataset done")
                        set_loaders[i] = iter(set_input)

                for (set_num,
                        set_name,
                        set_input,
                        set_writer,
                        set_log_freq,
                        set_do_backprop,
                        set_dict,
                        set_loader,
                        set_fake
                        ) in zip(
                        set_nums,
                        set_names,
                        set_inputs,
                        set_writers,
                        set_log_freqs,
                        set_do_backprops,
                        set_dicts,
                        set_loaders,
                        set_fakes
                        ):   

                    log_this = np.mod(step, set_log_freq)==0
                    total_time, read_time, iter_time = 0.0, 0.0, 0.0
                    if hyp.do_match_det:
                        if set_name == "test":
                            break
                    if log_this or set_do_backprop or hyp.break_constraint:
                        # st()
                        if log_this and set_name == "val":
                            halucinate_steps = hyp.halucinate_vals
                        else:
                            halucinate_steps = 1

                        for hal_num in range(halucinate_steps):
                            # st()
                            # print('%s: set_num %d; log_this %d; set_do_backprop %d; ' % (set_name, set_num, log_this, set_do_backprop))
                            # print('log_this = %s' % log_this)
                            # print('set_do_backprop = %s' % set_do_backprop)
                            if hal_num >0:
                                hyp.set_fake = True
                            read_start_time = time.time()

                            if hyp.debug_match or hyp.do_match_det:
                                hyp.B = 1                            
                            # feed = set_input[step]
                            # feed_cuda = {}
                            # for k in feed:
                            #     feed_cuda[k] = feed[k].to(self.device)
                            
                            feed = next(set_loader)

                            # st()
                            feed_cuda = {}
                            # st()
                            if hyp.do_clevr_sta:
                                tree_seq_filename = feed.pop('tree_seq_filename')
                                filename_e = feed.pop('filename_e')
                                filename_g = feed.pop('filename_g')
                            # st()
                            # st()
                            if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix" or hyp.dataset_name == "clevr_vqa"  or hyp.dataset_name == "carla_det":
                                if hyp.debug_match or hyp.do_match_det:
                                        feed['classes'] = np.array(feed['classes']).reshape([hyp.B*2,hyp.N])                                
                                classes = feed.pop('classes')

                            for k in feed:
                                #feed_cuda[k] = feed[k].to(self.device)
                                if hyp.typeVal == "content" or hyp.debug_match or hyp.do_match_det:
                                    feed_cuda[k] = feed[k].cuda(non_blocking=True).float().squeeze(0)
                                else:
                                    feed_cuda[k] = feed[k].cuda(non_blocking=True).float()

                            # feed_cuda = next(iter(set_input))
                            read_time = time.time() - read_start_time
                            # st()

                            if hyp.do_clevr_sta:
                                if hyp.typeVal == "content" or hyp.debug_match or hyp.do_match_det:
                                    feed_cuda['tree_seq_filename'] = np.array(tree_seq_filename).squeeze(1)
                                else:
                                    feed_cuda['tree_seq_filename'] = tree_seq_filename
                                feed_cuda['filename_e'] = filename_e
                                feed_cuda['filename_g'] = filename_g
                            
                            if hyp.dataset_name == "replica" or hyp.dataset_name == "carla_mix" or hyp.dataset_name == "clevr_vqa"   or hyp.dataset_name == "carla_det":
                                if not hyp.debug_match and not hyp.do_match_det:
                                    classes = np.transpose(np.array(classes))
                                feed_cuda['classes'] = classes
                                feed_cuda['filename_e'] = filename_e
                                feed_cuda['filename_g'] = filename_g                            

                            feed_cuda['writer'] = set_writer
                            feed_cuda['global_step'] = step
                            feed_cuda['set_num'] = set_num
                            feed_cuda['set_name'] = set_name
                            iter_start_time = time.time()

                            if hyp.typeVal == "content" or hyp.debug_match or hyp.do_match_det:
                                hyp.B = hyp.B*2
                            if set_do_backprop:
                                start_time =  time.time()                            
                                self.model.train()
                                # st()
                                loss, results = self.model(feed_cuda)
                                if hyp.profile_time:
                                    print("forwardpass time",time.time()-start_time)
                            else:
                                self.model.eval()
                                with torch.no_grad():
                                    loss, results = self.model(feed_cuda)
                                if hyp.halucinate_vals !=1:
                                    if hal_num == 0:
                                        maps = []
                                        filenames = []

                                    maps.append(results['maps'])
                                    filenames.append(results['filenames'])

                                    if (hal_num+1) == (hyp.halucinate_vals):
                                        maps_avg = np.mean(np.stack(maps),axis=0)
                                        for ind, overlap in enumerate(results['ious']):
                                            results['summ'].summ_scalar('ap_avg/%.2f_iou' % overlap, maps_avg[ind])



                            if hyp.typeVal == "content":
                                hyp.B = hyp.B//2

                            if hyp.online_cluster:
                                if step > hyp.cluster_iters:
                                    np.save(f'vqvae/cluster_centers_{hyp.object_quantize_dictsize}_iter{hyp.cluster_iters}.npy',
                                            self.model.kmeans.cluster_centers_)
                                    print("cluster saved")
                                    st()
                            if hyp.offline_cluster:
                                emb3D_e = results['emb3D_e']
                                emb3D_g = results['emb3D_g']
                                classes = results['classes']
                                if hyp.cluster_vox:
                                    vox_emb, classes = utils_eval.subsample_embs_voxs(emb3D_e, emb3D_g, classes=classes)
                                    _,F_DIM,NUM = list(vox_emb.shape)
                                    classes = np.expand_dims(classes,1)
                                    classes = np.repeat(classes,NUM,1).reshape([-1])
                                    vox_emb_flat = vox_emb.permute([0,2,1]).reshape([-1,F_DIM])
                                else:
                                    # emb3D_e_small = torch.nn.functional.interpolate(emb3D_e,size=[2,2,2],mode='trilinear')
                                    # emb3D_g_small = torch.nn.functional.interpolate(emb3D_g,size=[2,2,2],mode='trilinear')
                                    # st()
                                    c_B = emb3D_e.shape[0]
                                    emb3D_e_flat = emb3D_e.reshape(c_B,-1)
                                    emb3D_g_flat = emb3D_g.reshape(c_B,-1)
                                    emb3D_flat = emb3D_e_flat + emb3D_g_flat/2
                                    # emb_flat = torch.cat([emb3D_e_flat,emb3D_g_flat],dim=0)
                                    # classes = classes
                                emb_flat_np = emb3D_flat.cpu().numpy()
                                self.cluster_pool.update(emb_flat_np, classes)
                                if self.cluster_pool.is_full():
                                    embeds,classes = self.cluster_pool.fetch()
                                    kmeans = self.model.kmeans.fit(embeds)
                                    if "compute" in socket.gethostname():
                                        np.save(f'offline_obj_cluster/{hyp.feat_init}_cluster_centers_{hyp.object_quantize_dictsize}.npy',
                                                kmeans.cluster_centers_)                            
                                        pickle.dump(self.model.kmeans,open(f'offline_obj_cluster/{hyp.feat_init}_kmeans.p','wb'))
                                    self.cluster_pool.empty()
                                    # with open("offline_cluster" + '/%st.txt' % 'classes', 'w') as f:
                                    #     for index,embed in enumerate(classes):
                                    #         class_val = classes[index]
                                    #         f.write("%s\n" % class_val)
                                    # f.close()
                                    # with open("offline_cluster" + '/%st.txt' % 'embeddings', 'w') as f:
                                    #     N = len(embed)
                                    #     for index,embed in enumerate(embeds):
                                    #         print("writing {}/{} embed".format(index,N))
                                    #         embed_l_s = [str(i) for i in embed.tolist()]
                                    #         embed_str = '\t'.join(embed_l_s)
                                    #         f.write("%s\n" % embed_str)
                                    #     st()
                                    # f.close()

                                    # embed = embeds[index]
                            if hyp.offline_cluster_eval:
                                emb3D_e = results['emb3D_e']
                                emb3D_g = results['emb3D_g']
                                classes = results['classes']
                                if hyp.use_kmeans:
                                    emb3D_e_flat = emb3D_e.reshape(hyp.B,-1)
                                    emb3D_g_flat = emb3D_g.reshape(hyp.B,-1)
                                    emb3D_e_flat = emb3D_e_flat.cpu().numpy()
                                    emb3D_g_flat = emb3D_g_flat.cpu().numpy()
                                    e_indexes = self.model.kmeans.predict(emb3D_e_flat)
                                    g_indexes = self.model.kmeans.predict(emb3D_g_flat)
                                else:
                                    e_indexes = self.model.quantizer.predict(emb3D_e)
                                    g_indexes = self.model.quantizer.predict(emb3D_g)
                                for index in range(hyp.B):
                                    class_val = classes[index]
                                    e_i = e_indexes[index]
                                    g_i = g_indexes[index]
                                    self.info_dict[str(e_i)].append(class_val)
                                    self.info_dict[str(g_i)].append(class_val)

                                if (step % 1000) == 0:
                                    scores_dict = {}
                                    most_freq_dict = {}
                                    scores_list = []
                                    for key,item in self.info_dict.items():
                                        most_freq_word = utils_basic.most_frequent(item)
                                        mismatch = 0 
                                        for i in item:
                                            if i != most_freq_word:
                                                mismatch += 1
                                        precision = float(len(item)- mismatch)/len(item)
                                        scores_dict[key] = precision
                                        most_freq_dict[key] = most_freq_word
                                        scores_list.append(precision)
                                    print(np.mean(scores_list))
                                    if step == 1000:
                                        st()
                                    # st()
                                    print("evaluate")

                            if set_do_backprop and hyp.moc:
                                self.model_key.eval()
                                with torch.no_grad():                               
                                    loss_key, results_key = self.model_key(feed_cuda)

                                emb3D_e = results['emb3D_e']
                                emb3D_g_key = results_key['emb3D_g']

                                ml_loss_3d = self.ml_loss(emb3D_e,emb3D_g_key,self.poolvox_moc)
                                loss += ml_loss_3d

                                classes_key_t = results_key['classes']
                        
                                emb3D_e_key_t = results_key['emb3D_e']
                                emb3D_g_key_t = results_key['emb3D_g']
                        
                                vox_emb_key_t, classes_key_t = utils_eval.subsample_embs_voxs(emb3D_e_key_t, emb3D_g_key_t, classes= classes_key_t)
                                
                                self.poolvox_moc.update(vox_emb_key_t,classes_key_t)

                                if hyp.moc_2d:
                                    emb2D_e = results['emb2D_e']
                                    emb2D_g_key = results_key['emb2D_g']
                                        
                                    ml_loss_2d = self.ml_loss(emb2D_e,emb2D_g_key,self.poolvox_moc)
                                    loss += ml_loss_2d

                                
                                
                                


                            # st()

                            loss_vis = loss.cpu().item()
                            # st()    
                            summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                                   global_step=feed_cuda['global_step'],
                                                   set_name=feed_cuda['set_name'],
                                                   fps=8)
                            summ_writer.summ_scalar('loss',loss_vis)
                            
                            if (hyp.do_eval_recall):
                                # (2, 32, 128, 128)
                                set_dict = self.eval_dicts["eval"]
                                # (2, 3, 256, 256)
                                rgb = results['rgb'].cpu().detach().numpy()
                                rgb = np.transpose(rgb, (0, 2, 3, 1))
                                # if hyp.do_emb2D:
                                #     emb2D_e = results['emb2D_e'].cpu().detach().numpy()
                                #     emb2D_e = np.transpose(emb2D_e, (0, 2, 3, 1))
                                #     # (2, 32, 128, 128)
                                #     emb2D_g = results['emb2D_g'].cpu().detach().numpy()
                                #     emb2D_g = np.transpose(emb2D_g, (0, 2, 3, 1))
                                #     samps = 100 if set_dict['pool2D_e'].is_full() else 100
                                #     emb2D_e, emb2D_g, rgb = utils_eval.subsample_embs_2D(emb2D_e, emb2D_g, rgb, samps=samps)
                                #     set_dict['pool2D_e'].update(emb2D_e, rgb)
                                #     set_dict['pool2D_g'].update(emb2D_g, rgb)

                                # (2, 3, 96, 48, 96)
                                if hyp.hard_vis:
                                    visual3D = results['visual3D'].cpu().detach().numpy()
                                    temp_visual3D = np.transpose(visual3D, (0, 2, 3, 1))                            
                                    classes = results['classes']

                                if hyp.eval_recall_o:
                                    visual3D = rgb
                                    classes = results['classes']
                                else:                        
                                    visual3D = results['visual3D'].cpu().detach().numpy()
                                    visual3D = np.transpose(visual3D, (0, 2, 3, 4, 1))
                                # emb3D_e = results['emb3D_e'].cpu().detach().numpy()
                                # emb3D_e = np.transpose(emb3D_e, (0, 2, 3, 4, 1))
                                # emb3D_g = results['emb3D_g'].cpu().detach().numpy()
                                # emb3D_g = np.transpose(emb3D_g, (0, 2, 3, 4, 1))
                                emb3D_e = results['emb3D_e'].detach()
                                emb3D_g = results['emb3D_g'].detach()
                                valid3D = results['valid3D']
                                visual2D = results['rgb'].detach()
                                # valid3D = results['valid3D'].cpu().detach().numpy()
                                # valid3D = np.transpose(valid3D, (0, 2, 3, 4, 1))
                                # earlier i had 10 here, but with infrequent logging, the plot is inacc
                                samps = 100 if set_dict['pool3D_e'].is_full() else 100
                                # if hyp.eval_recall_o:
                                #     # emb3D_e, emb3D_g, visual3D, classes = utils_eval.subsample_embs_3D_o(emb3D_e, emb3D_g, valid3D, visual3D, classes= classes,samps=samps)
                                #     emb3D_e, emb3D_g, visual3D, classes,_ = utils_eval.subsample_embs_3D_o_cuda(emb3D_e, emb3D_g, valid3D, visual3D, classes= classes,filenames=classes)
                                # else:
                                #     emb3D_e, emb3D_g, visual3D = utils_eval.subsample_embs_3D(emb3D_e, emb3D_g, valid3D, visual3D, samps=samps)

                                if isinstance(visual3D,type(torch.tensor([]))):
                                    visual3D = visual3D.cpu().detach().numpy()
                                    visual3D = np.transpose(visual3D, (0, 2, 3, 1))
                                # st()
                                if hyp.eval_recall_o:
                                    if hyp.hard_vis:
                                        set_dict['pool3D_e'].update(emb3D_e, temp_visual3D, classes, visual2D)
                                        set_dict['pool3D_g'].update(emb3D_g, temp_visual3D, classes, visual2D)                                
                                    else:
                                        if hyp.cpu:
                                            set_dict['pool3D_e'].update(emb3D_e.cpu(), visual3D, classes, visual2D)
                                            set_dict['pool3D_g'].update(emb3D_g.cpu(), visual3D, classes, visual2D)
                                        else:
                                            set_dict['pool3D_e'].update(emb3D_e, visual3D, classes, visual2D)
                                            set_dict['pool3D_g'].update(emb3D_g, visual3D, classes, visual2D)
                                else:                        
                                    set_dict['pool3D_e'].update(emb3D_e, visual3D)
                                    set_dict['pool3D_g'].update(emb3D_g, visual3D)
                                
                                # if hyp.do_emb2D:
                                #     set_dict['precision2D'], set_dict['neighbors2D'] = utils_eval.compute_precision(
                                #         set_dict['pool2D_e'].fetch(),
                                #         set_dict['pool2D_g'].fetch(),
                                #         recalls=self.recalls,
                                #         pool_size=self.pool_size)
                                # st()
                                if hyp.eval_recall_o:
                                    set_dict['precision3D'], set_dict['neighbors3D'],ranks, exp_done,filenames  = utils_eval.compute_precision_o_cuda(
                                        set_dict['pool3D_e'],
                                        set_dict['pool3D_g'],
                                        hyp.eval_compute_freq,
                                        self.hpm,
                                        self.mbr16,
                                        recalls=self.recalls,
                                        pool_size=self.pool_size,
                                        steps_done=step,
                                        summ_writer=summ_writer,
                                        mbr_unpr=self.mbr_unpr)                            
                                    # set_dict['precision3D'], set_dict['neighbors3D'] = utils_eval.compute_precision_o(
                                    #     set_dict['pool3D_e'].fetch(),
                                    #     set_dict['pool3D_g'].fetch(),
                                    #     recalls=self.recalls,
                                    #     pool_size=self.pool_size,
                                    #     summ_writer=summ_writer)
                                else:
                                    set_dict['precision3D'], set_dict['neighbors3D'] = utils_eval.compute_precision(
                                        set_dict['pool3D_e'].fetch(),
                                        set_dict['pool3D_g'].fetch(),
                                        recalls=self.recalls,
                                        pool_size=self.pool_size)                            
                                
                                ns = "retrieval/"

                                # if hyp.do_emb2D:
                                #     precision2D = set_dict['precision2D']
                                
                                precision3D = set_dict['precision3D']

                                # if hyp.do_emb2D:
                                #     neighbors2D = torch.from_numpy(set_dict['neighbors2D']).float().permute(2,0,1)
                                
                                neighbors3D = torch.from_numpy(set_dict['neighbors3D']).float().permute(2,0,1)

                                
                                # if hyp.do_emb2D:
                                #     summ_writer.summ_scalar(ns +'precision2D_01', precision2D[0])
                                #     summ_writer.summ_scalar(ns +'precision2D_05', precision2D[1])
                                #     summ_writer.summ_scalar(ns +'precision2D_10', precision2D[2])
                                #     summ_writer.summ_rgb(ns +'neighbors2D', utils_improc.preprocess_color(neighbors2D).unsqueeze(0))
                                # 3D
                                if hyp.eval_recall_o:
                                    for key,precisions in precision3D.items():
                                        if "average" in precisions:
                                            average = precisions.pop("average")
                                            summ_writer.summ_scalar(ns + 'precision3D_{:02d}_avg'.format(int(key)),average)
                                        if hyp.eval_recall_summ_o:
                                            summ_writer.summ_scalars(ns + 'precision3D_{:02d}'.format(int(key)),dict(precisions))
                                else:
                                    summ_writer.summ_scalar(ns + 'precision3D_01', precision3D[0])
                                    summ_writer.summ_scalar(ns + 'precision3D_05', precision3D[1])
                                    summ_writer.summ_scalar(ns + 'precision3D_10', precision3D[2])
                                
                                summ_writer.summ_rgb(ns + 'neighbors3D', utils_improc.preprocess_color(neighbors3D).unsqueeze(0))
               
                            if set_do_backprop and hyp.sudo_backprop:
                                if hyp.accumulate_grad:
                                    loss = loss / hyp.accumulation_steps                # Normalize our loss (if averaged)
                                    loss.backward()                                 # Backward pass
                                    if (step) % hyp.accumulation_steps == 0:             # Wait for several backward steps
                                        self.optimizer.step()                            # Now we can do an optimizer step
                                        self.optimizer.zero_grad()
                                else:
                                    self.optimizer.zero_grad()
                                    loss.backward()
                                    self.optimizer.step()

                            if hyp.moc:
                                self.momentum_update(self.model,self.model_key)



                            hyp.sudo_backprop = True

                            iter_time = time.time()-iter_start_time
                            total_time = time.time()-self.start_time
                            # st()

                            if hyp.eval_recall_o:
                                print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Object_Pool: %4d/%4d; (%s)" % (hyp.name,
                                                                                                    step,
                                                                                                    hyp.max_iters,
                                                                                                    total_time,
                                                                                                    read_time,
                                                                                                    iter_time,
                                                                                                    loss_vis,
                                                                                                    self.eval_dicts["eval"]['pool3D_e'].num,
                                                                                                    self.pool_size,
                                                                                                    set_name))
                            elif hyp.offline_cluster:
                                print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Cluster_Pool: %4d/%4d; (%s)" % (hyp.name,
                                                                                                    step,
                                                                                                    hyp.max_iters,
                                                                                                    total_time,
                                                                                                    read_time,
                                                                                                    iter_time,
                                                                                                    loss_vis,
                                                                                                    self.cluster_pool.num,
                                                                                                    self.cluster_pool.pool_size,
                                                                                                    set_name))                        
                            elif hyp.online_cluster:
                                print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f; Cluster_Pool: %4d/%4d; (%s)" % (hyp.name,
                                                                                                    step,
                                                                                                    hyp.max_iters,
                                                                                                    total_time,
                                                                                                    read_time,
                                                                                                    iter_time,
                                                                                                    loss_vis,
                                                                                                    self.model.voxel_queue.num,
                                                                                                    self.model.voxel_queue.pool_size,
                                                                                                    set_name))                                                
                            else:
                                print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                                    step,
                                                                                                    hyp.max_iters,
                                                                                                    total_time,
                                                                                                    read_time,
                                                                                                    iter_time,
                                                                                                    loss_vis,
                                                                                                    set_name))
                if np.mod(step, hyp.snap_freq) == 0:
                    saverloader.save(self.model, self.checkpoint_dir, step, self.optimizer)

            for writer in set_writers: #close writers to flush cache into file
                writer.close()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
