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



    def infer(self):
        pass
    def momentum_update(self,model_q, model_k, beta = 0.999):
        param_k = model_k.state_dict()
        param_q = model_q.named_parameters()
        for n, q in param_q:
            if n in param_k:
                param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
        model_k.load_state_dict(param_k)



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

        set_nums = []
        set_names = []
        set_inputs = []
        set_writers = []
        set_log_freqs = []
        set_do_backprops = []
        set_dicts = []
        set_loaders = []


        # st()
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

            # st()

        for step in range(self.start_iter+1, hyp.max_iters+1):
            for i, (set_input) in enumerate(set_inputs):
                if step % len(set_input) == 0: #restart after one epoch. Note this does nothing for the tfrecord loader
                    set_loaders[i] = iter(set_input)

            for (set_num,
                    set_name,
                    set_input,
                    set_writer,
                    set_log_freq,
                    set_do_backprop,
                    set_dict,
                    set_loader,
                    ) in zip(
                    set_nums,
                    set_names,
                    set_inputs,
                    set_writers,
                    set_log_freqs,
                    set_do_backprops,
                    set_dicts,
                    set_loaders,
                    ):   

                log_this = np.mod(step, set_log_freq)==0
                total_time, read_time, iter_time = 0.0, 0.0, 0.0

                if log_this or set_do_backprop or hyp.break_constraint:
                    read_start_time = time.time()

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



                    loss_vis = loss.cpu().item()

                    summ_writer = utils_improc.Summ_writer(writer=feed_cuda['writer'],
                                           global_step=feed_cuda['global_step'],
                                           set_name=feed_cuda['set_name'],
                                           fps=8)
                    summ_writer.summ_scalar('loss',loss_vis)
                    

       
                    if set_do_backprop:
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



                    iter_time = time.time()-iter_start_time
                    total_time = time.time()-self.start_time

                    print("%s; [%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f (%s)" % (hyp.name,
                                                                                        step,
                                                                                        hyp.max_iters,
                                                                                        total_time,
                                                                                        read_time,
                                                                                        iter_time,
                                                                                        loss_vis,
                                                                                        set_name))
                # st()
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
