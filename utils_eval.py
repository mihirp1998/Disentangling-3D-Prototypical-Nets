import io as sysio
import time
import hyperparams as hyp
import random
import numba
import numpy as np
import ipdb 
from collections import defaultdict
from sklearn.preprocessing import normalize
import utils_box
import utils_ap
import utils_basic
import copy
import torch
import cross_corr
import utils_geom
st = ipdb.set_trace
EPS = 1e-6

def makeDict(annos):
    assert annos.ndim == 3 and annos.shape[2] == 9
    annos = [{'location'   : a[:,0:3],
              'dimensions' : a[:,3:6],
              'rotation_y' : a[:,6]}
             for a in annos]
    return annos
    
def make_border_green(vis):
    vis = np.copy(vis)
    vis[0,:,0] = 0
    vis[0,:,1] = 255
    vis[0,:,2] = 0
    
    vis[-1,:,0] = 0
    vis[-1,:,1] = 255
    vis[-1,:,2] = 0

    vis[:,0,0] = 0
    vis[:,0,1] = 255
    vis[:,0,2] = 0
    
    vis[:,-1,0] = 0
    vis[:,-1,1] = 255
    vis[:,-1,2] = 0
    return vis

def drop_invalid_boxes(boxlist_e, boxlist_g, scorelist_e, scorelist_g):
    # print('before:')
    # print(boxlist_e.shape)
    # print(boxlist_g.shape)
    boxlist_e_, boxlist_g_, scorelist_e_, scorelist_g_ = [], [], [], []
    for i in list(range(len(boxlist_e))):
        box_e = boxlist_e[i]
        # print('box_e', box_e)
        score_e = scorelist_e[i]
        valid_e = np.where(box_e[:,3] > 0.0) # lx
        boxlist_e_.append(box_e[valid_e])
        scorelist_e_.append(score_e[valid_e])
    # print('boxlist_e_', boxlist_e_)
    for i in list(range(len(boxlist_g))):
        box_g = boxlist_g[i]
        score_g = scorelist_g[i]
        valid_g = np.where(score_g > 0.5)
        boxlist_g_.append(box_g[valid_g])
        scorelist_g_.append(score_g[valid_g])
    # print('boxlist_g_', boxlist_g_)
    boxlist_e, boxlist_g, scorelist_e, scorelist_g = np.array(boxlist_e_), np.array(boxlist_g_), np.array(scorelist_e_), np.array(scorelist_g_)
    return boxlist_e, boxlist_g, scorelist_e, scorelist_g


def make_border_blue(vis):
    vis = np.copy(vis)
    vis[0,:,0] = 0
    vis[0,:,1] = 0
    vis[0,:,2] = 255
    
    vis[-1,:,0] = 0
    vis[-1,:,1] = 0
    vis[-1,:,2] = 255

    vis[:,0,0] = 0
    vis[:,0,1] = 0
    vis[:,0,2] = 255
    
    vis[:,-1,0] = 0
    vis[:,-1,1] = 0
    vis[:,-1,2] = 255
    return vis

def make_border_black(vis):
    vis = np.copy(vis)
    vis[0,:,:] = 0
    vis[-1,:,:] = 0
    vis[:,0,:] = 0
    vis[:,-1,:] = 0
    return vis

def compute_precision(xxx_todo_changeme, xxx_todo_changeme1, recalls=[1,3,5], pool_size=100):
    # inputs are lists
    # list elements are H x W x C
    
    (emb_e, vis_e) = xxx_todo_changeme
    (emb_g, vis_g) = xxx_todo_changeme1
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    precision = np.zeros(len(recalls), np.float32)
    # print 'precision B = %d' % B
    # st()
    if len(vis_e[0].shape)==4:
        # H x W x D x C
        # squish the height dim, and look at the birdview
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) # vis_e shape is weird

    perm = np.random.permutation(B)
    vis_inds = perm[:10] # just vis 10 queries
    
    # print 'B = %d; pool_size = %d' % (B, pool_size)
    
    if B >= pool_size: # otherwise it's not going to be accurate
        emb_e = np.stack(emb_e, axis=0)
        emb_g = np.stack(emb_g, axis=0)
        # emb_e = np.concatenate(emb_e, axis=0)
        # emb_g = np.concatenate(emb_g, axis=0)
        vect_e = normalize(np.reshape(emb_e, [B, -1]))
        vect_g = normalize(np.reshape(emb_g, [B, -1]))
        scores = np.dot(vect_e, np.transpose(vect_g))

        ranks = np.flip(np.argsort(scores), axis=1)

        vis = []
        
        # going over each query

        for i in vis_inds:
            minivis = []
            # first col: query
            # minivis.append(vis_e[i])
            minivis.append(make_border_black(vis_e[i]))
            
            # # second col: true answer
            # minivis.append(vis_g[i])
            
            # remaining cols: ranked answers
            for j in list(range(10)):
                v = vis_g[ranks[i, j]]
                if ranks[i, j]==i:
                    minivis.append(make_border_green(v))
                else:
                    minivis.append(v)
            # concat retrievals along width
            minivis = np.concatenate(minivis, axis=1)
            # print 'got this minivis:', 
            # print minivis.shape
            
            vis.append(minivis)
        # concat examples along height
        vis = np.concatenate(vis, axis=0)
        # print 'got this vis:', 
        # print vis.shape
            
        for recall_id, recall in enumerate(recalls):
            for emb_id in list(range(B)):
                if emb_id in ranks[emb_id, :recall]:
                    precision[recall_id] += 1
            # print("precision@", recall, float(precision[recall_id])/float(B))
        precision = precision/float(B)
    else:
        precision = np.nan*precision
        vis = np.zeros((H*10, W*11, 3), np.uint8)
    # print 'precision  %.2f' % np.mean(precision)
    
    return precision, vis

def get_negative_samples(queue_dict,class_val):
    negative_samples = []
    for key,val in queue_dict.items():
        if key != class_val:
            negative_samples.append(torch.stack(val.fetch()))
    negative_samples = torch.cat(negative_samples,dim=0)
    return negative_samples


def subsample_embs_voxs(emb3DS_e, emb3DS_g, classes):
    emb3Ds = []
    # st()
    for index, emb3D_e in enumerate(emb3DS_e):
        emb3D_e_flat = emb3D_e.reshape([hyp.feat_dim,-1])
        emb3D_g_flat = emb3DS_g[index].reshape([hyp.feat_dim,-1])
        _,dim = list(emb3D_g_flat.shape)
        index_to_use = np.random.permutation(dim)[:hyp.emb_moc.indexes_to_take]
        emb3D_e_flat_filtered  = emb3D_e_flat[:,index_to_use]
        index_to_use = np.random.permutation(dim)[:hyp.emb_moc.indexes_to_take]
        emb3D_g_flat_filtered  =  emb3D_g_flat[:,index_to_use]
        emb3D = torch.cat([emb3D_e_flat_filtered,emb3D_g_flat_filtered],dim=1)
        emb3D = emb3D[:,np.random.permutation(hyp.emb_moc.indexes_to_take*2)]
        emb3Ds.append(emb3D)
    emb3Ds = torch.stack(emb3Ds)
    return emb3Ds,classes



def subsample_embs_voxs_positive(emb3DS_e,emb3DS_g_key, classes):
    emb3Ds = []
    emb3Ds_key = []
    for index, emb3D_e in enumerate(emb3DS_e):
        emb3D_e_flat = emb3D_e.reshape([hyp.feat_dim,-1])
        emb3D_g_key_flat = emb3DS_g_key[index].reshape([hyp.feat_dim,-1])
        _,dim = list(emb3D_e_flat.shape)
        index_to_use = np.random.permutation(dim)[:hyp.emb_moc.indexes_to_take*2]

        emb3D_e_flat_filtered  = emb3D_e_flat[:,index_to_use]
        emb3D_g_key_flat_filtered  =  emb3D_g_key_flat[:,index_to_use]

        emb3Ds.append(emb3D_e_flat_filtered)
        emb3Ds_key.append(emb3D_g_key_flat_filtered)
    emb3Ds = torch.stack(emb3Ds)
    emb3Ds_key = torch.stack(emb3Ds_key)
    return emb3Ds,emb3Ds_key,classes


def compute_precision_o(xxx_todo_changeme, xxx_todo_changeme1, recalls=[1,3,5], pool_size=100, summ_writer=None):
    # inputs are lists
    # list elements are H x W x C
    # st()
    (emb_e, vis_e, class_e) = xxx_todo_changeme
    (emb_g, vis_g, class_g) = xxx_todo_changeme1
    
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    precision = lambda  : 0
    precision_per_recall = {}
    # st()
    # if summ_writer != None:
    #     emb_g_0 = torch.from_numpy(emb_g[-1])
    #     emb_g_to_visualize = emb_g_0.permute(3, 0, 1, 2).unsqueeze(0)
    #     if hyp.debug_eval_recall_o:
    #         summ_writer.summ_occ("precision_o_emb_g", emb_g_to_visualize.cuda())
    #     else:
    #         summ_writer.summ_feat("precision_o_emb_g", emb_g_to_visualize)
    for recall in recalls:
        precision_per_object = defaultdict(precision)
        precision_per_recall[str(recall)] = copy.deepcopy(precision_per_object)

    instances_per_object = defaultdict(lambda: 0)

    # print 'precision B = %d' % B
    # st()
    if len(vis_e[0].shape)==4:
        # H x W x D x C
        # squish the height dim, and look at the birdview
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) # vis_e shape is weird

    perm = np.random.permutation(B)
    perm = list(range(10))
    vis_inds = perm[:10] # just vis 10 queries
    
    # print 'B = %d; pool_size = %d' % (B, pool_size)
    
    if B >= pool_size: # otherwise it's not going to be accurate
        emb_e = np.stack(emb_e, axis=0)
        emb_g = np.stack(emb_g, axis=0)
        # emb_e = np.concatenate(emb_e, axis=0)
        # emb_g = np.concatenate(emb_g, axis=0)
        vect_e = normalize(np.reshape(emb_e, [B, -1]))
        vect_g = normalize(np.reshape(emb_g, [B, -1]))
        if hyp.do_orientation:
            scores = cross_corr.orient_and_calculate_scores(emb_e, emb_g)
            scores = scores.cpu().detach().numpy()
        else:
            scores = np.dot(vect_e, np.transpose(vect_g))
            # st()
        # argsort_score = np.argsort(scores)
        ranks = np.flip(np.argsort(scores), axis=1)

        vis = []
        
        # going over each query

        for i in vis_inds:
            minivis = []
            class_to_query = class_e[i]
            # miniclass = []
            # first col: query
            # minivis.append(vis_e[i])
            minivis.append(make_border_black(vis_e[i]))
            # miniclass.append()

            
            # # second col: true answer
            # minivis.append(vis_g[i])
            
            # remaining cols: ranked answers
            for j in list(range(10)):
                v = vis_g[ranks[i, j]]
                class_atRank_j = class_g[ranks[i, j]]
                if ranks[i, j]==i:
                    minivis.append(make_border_green(v))
                elif class_atRank_j == class_to_query:
                    minivis.append(make_border_blue(v))                    
                else:
                    minivis.append(v)
            # concat retrievals along width
            minivis = np.concatenate(minivis, axis=1)
            # print 'got this minivis:', 
            # print minivis.shape
            
            vis.append(minivis)
        # concat examples along height
        vis = np.concatenate(vis, axis=0)
        # print 'got this vis:', 
        # print vis.shape
        # st()
        for recall_id, recall in enumerate(recalls):
            for query_id in list(range(B)):
                class_to_query = class_e[query_id]
                if recall_id == 0:
                    # calculate only the first time
                    instances_per_object[class_to_query] += 1
                for retrieved_id in ranks[query_id, :recall]:
                    class_retrieved = class_g[retrieved_id]
                    if class_to_query == class_retrieved:
                        precision_per_recall[str(recall)][class_to_query] += 1
            # print("precision@", recall, float(precision[recall_id])/float(B))
        # some postprocessing to do
        # st()
        for recall_id, recall in enumerate(recalls):
            total = 0
            for key , precision in precision_per_recall[str(recall)].items():
            # a ste p to make sure that there are no wrong precisions
                # if  instances_per_object[key] < recall:
                #     precision = np.nan
                # normalization per instance
                precision = precision/instances_per_object[key]
                # normalization recall score
                precision = precision/recall

                total += precision
                
                precision_per_recall[str(recall)][key] = precision
            
            if len(precision_per_recall[str(recall)].keys()) == 0:
                average = 0
                print("precission per recall has no keys for recall value: ", str(recall))
            else:
                average = total/len(precision_per_recall[str(recall)].keys())
            precision_per_recall[str(recall)]['average'] =  average
    else:
        vis = np.zeros((H*10, W*11, 3), np.uint8)
        for recall in recalls:
            for key , precision in precision_per_recall[str(recall)].items():
                precision_per_recall[str(recall)][key] = np.nan
    # print 'precision  %.2f' % np.mean(precision)
    return precision_per_recall, vis


def compute_patch_based_scores(pool_e, pool_g, num_embeds,hpm):
    # hpm = hardPositiveMiner.HardPositiveMiner()

    num_patches_per_emb = hyp.max.num_patches_per_emb
    scores = torch.zeros((num_embeds, num_embeds)).cuda()
    '''
    This will create a dummy rank matrix which will look like this:
    0 1 2 ... num_embeds (1st row)
    0 1 2 ... num_embeds (2nd row)
    .
    .
    0 1 2 ... num_embeds  (num_embeds th row)
    '''
    dummy_ranks, _ = np.meshgrid(np.arange(num_embeds), np.arange(num_embeds))
    # _, dummy_ranks = torch.meshgrid(torch.arange(num_embeds),torch.arange(num_embeds))
    # dummy_ranks = dummy_ranks.cuda()
    # import time
    # st()
    # start_time = time.time()
    for h_init in list(range(4,16,4)):
        for d_init in list(range(4,16,4)):
            for w_init in list(range(4,16,4)):
                featQuery_i , perm_i =  hpm.extractPatches_det(pool_e,d_init,h_init,w_init)
                topkImg_i, _, topkValue_i, _, _, _,_  = hpm.RetrievalResForExpectation(pool_g, featQuery_i)
                for j in range(topkImg_i.shape[0]):
                    scores[j, topkImg_i[j].long()] += topkValue_i[j].cuda()
    # print("time for eval",time.time() - start_time)
    # st()
    return scores

def compute_patch_based_vis(pool_e, pool_g, num_embeds,summ_writer,hpm):
    # hpm = hardPositiveMiner.HardPositiveMiner()

    num_patches_per_emb = hyp.max.num_patches_per_emb
    scores = torch.zeros((num_embeds, num_embeds)).cuda()
    '''
    This will create a dummy rank matrix which will look like this:
    0 1 2 ... num_embeds (1st row)
    0 1 2 ... num_embeds (2nd row)
    .
    .
    0 1 2 ... num_embeds  (num_embeds th row)
    '''
    # _, dummy_ranks = torch.meshgrid(torch.arange(num_embeds),torch.arange(num_embeds))
    # dummy_ranks = dummy_ranks.cuda()
    # import time
    # st()
    # start_time = time.time()
    h_init_e = np.random.randint(2,14)
    d_init_e = np.random.randint(2,14)
    w_init_e = np.random.randint(2,14)
    _, unps_e, classes_e, _,vis2Ds_e = pool_e.fetch()
    _, unps_g, classes_g, _ ,vis2Ds_g = pool_e.fetch()

    featQuery_i , perm_i =  hpm.extractPatches_det(pool_e,d_init_e,h_init_e,w_init_e)
    featQuery_i = featQuery_i[:1]
    
    unp_e = unps_e[0]
    vis2D_e = vis2Ds_e[0]
    topkImg_i, topkScale, topkValue_i, topkW , topkH , topkD, topkR  = hpm.RetrievalResForExpectation(pool_g, featQuery_i)
    # pool_g_index_retrieved = topkImg_i[0,0]
    
    # W_top_g = topkW[0,0]
    # H_top_g = topkH[0,0]
    # D_top_g = topkD[0,0]
    
    # unp_g_top = unps_g[pool_g_index_retrieved]
    summ_writer.summ_evalmines("eval_mines",[[topkImg_i,topkD,topkH,topkW,topkR],[d_init_e,h_init_e,w_init_e],[unps_g,unp_e],[vis2Ds_g,vis2D_e]],hpm.mbr_unpr)
    size = pool_g.num



def compute_precision_o_cuda(pool_e, pool_g,max_iters,hpm,mbr16, recalls=[1,3,5], pool_size=100, summ_writer=None,steps_done=0,mbr_unpr=None):
    (emb_e, vis_e, class_e, files_e,vis2D) = pool_e.fetch()
    (emb_g, vis_g, class_g, files_g,vis2D) = pool_g.fetch()
    # st()
    assert(len(emb_e)==len(emb_g))
    B = len(emb_e)
    precision = lambda  : 0
    precision_per_recall = {}
    if hyp.debug_eval_recall_o and summ_writer is not None:
        emb_g_0 = torch.from_numpy(emb_g[-1])
        emb_g_to_visualize = emb_g_0.permute(3, 0, 1, 2).unsqueeze(0)
        summ_writer.summ_occ("precision_o_emb_g", emb_g_to_visualize.cuda())

    
    for recall in recalls:
        precision_per_object = defaultdict(precision)
        precision_per_recall[str(recall)] = copy.deepcopy(precision_per_object)

    instances_per_object = defaultdict(lambda: 0)

    # print 'precision B = %d' % B
    # st()
    if len(vis_e[0].shape)==4:
        # H x W x D x C
        # squish the height dim, and look at the birdview
        vis_e = [np.mean(vis, axis=0) for vis in vis_e]
        vis_g = [np.mean(vis, axis=0) for vis in vis_g]
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    elif len(vis_e[0].shape)==3:
        # H x W x C
        H = vis_e[0].shape[0]
        W = vis_e[0].shape[1]
    else:
        assert(False) # vis_e shape is weird

    perm = np.random.permutation(B)
    # perm = list(range(10))
    vis_inds = perm[:10] # just vis 10 queries
    
    exp_done = False
    # print 'B = %d; pool_size = %d' % (B, pool_size)
    # if B >= pool_size:
    #     exp_done = True
    # st()
    if (B >= pool_size and (steps_done % max_iters == 0)): # otherwise it's not going to be accurate
        emb_e = torch.stack(emb_e, axis=0)
        emb_g = torch.stack(emb_g, axis=0)
        vect_e = torch.nn.functional.normalize(torch.reshape(emb_e, [B, -1]),dim=1)
        vect_g = torch.nn.functional.normalize(torch.reshape(emb_g, [B, -1]),dim=1)
        # st()
        if hyp.max.hardmining and not hyp.max.exceptions or hyp.hard_eval:
            # Right now assuming there is no rotational deformation.
            if hyp.hard_vis:
                if hyp.summ_all:
                    old = summ_writer.save_this
                    summ_writer.save_this = True
                compute_patch_based_vis(pool_e, pool_g, len(emb_e),summ_writer,hpm)
                scores = cross_corr.orient_and_calculate_scores_cuda(emb_e, emb_g, mbr16)
                summ_writer.summ_diff_orientation("target_bestR_query",pool_e, pool_g,scores,mbr16,mbr_unpr)
                if hyp.summ_all:
                    summ_writer.save_this = old
            else:
                scores = compute_patch_based_scores(pool_e, pool_g, len(emb_e),hpm)
        else:
            if hyp.do_orientation:
                scores = cross_corr.orient_and_calculate_scores_cuda(emb_e, emb_g, mbr16)
            else:
                scores = torch.matmul(vect_e, vect_g.t())
        if not hyp.hard_vis:
            scores = scores.cpu().detach().numpy()
            ranks = np.flip(np.argsort(scores), axis=1)

            vis = []
            # st()
            
            # going over each query

            for i in vis_inds:
                minivis = []
                class_to_query = class_e[i]
                # miniclass = []
                # first col: query
                # minivis.append(vis_e[i])
                minivis.append(make_border_black(vis_e[i]))
                # miniclass.append()
                # # second col: true answer
                # minivis.append(vis_g[i])
                # remaining cols: ranked answers
                for j in list(range(10)):
                    v = vis_g[ranks[i, j]]
                    class_atRank_j = class_g[ranks[i, j]]
                    if ranks[i, j]==i:
                        minivis.append(make_border_green(v))
                    elif class_atRank_j == class_to_query:
                        minivis.append(make_border_blue(v))
                    else:
                        minivis.append(v)
                # concat retrievals along width
                minivis = np.concatenate(minivis, axis=1)
                # print 'got this minivis:', 
                # print minivis.shape
                vis.append(minivis)
            # concat examples along height
            vis = np.concatenate(vis, axis=0)
            exp_done = True
            # print 'got this vis:', 
            # print vis.shape
            # st()
            for recall_id, recall in enumerate(recalls):
                for query_id in list(range(B)):
                    class_to_query = class_e[query_id]
                    if recall_id == 0:
                        # calculate only the first time
                        instances_per_object[class_to_query] += 1
                    for retrieved_id in ranks[query_id, :recall]:
                        class_retrieved = class_g[retrieved_id]
                        if class_to_query == class_retrieved:
                            precision_per_recall[str(recall)][class_to_query] += 1
                # print("precision@", recall, float(precision[recall_id])/float(B))
            # some postprocessing to do
            for recall_id, recall in enumerate(recalls):
                total = 0
                for key , precision in precision_per_recall[str(recall)].items():
                # a ste p to make sure that there are no wrong precisions
                    # if  instances_per_object[key] < recall:
                    #     precision = np.nan
                    # normalization per instance
                    precision = precision/instances_per_object[key]
                    # normalization recall score
                    precision = precision/recall

                    total += precision
                    
                    precision_per_recall[str(recall)][key] = precision
                
                if len(precision_per_recall[str(recall)].keys()) == 0:
                    average = 0
                    print("precission per recall has no keys for recall value: ", str(recall))
                else:
                    average = total/len(precision_per_recall[str(recall)].keys())
                precision_per_recall[str(recall)]['average'] =  average
        else:
            vis = np.zeros((H*10, W*11, 3), np.uint8)
            ranks = np.zeros((B,B), np.uint8)
            for recall in recalls:
                for key , precision in precision_per_recall[str(recall)].items():
                    precision_per_recall[str(recall)][key] = np.nan                
    else:
        vis = np.zeros((H*10, W*11, 3), np.uint8)
        ranks = np.zeros((B,B), np.uint8)
        for recall in recalls:
            for key , precision in precision_per_recall[str(recall)].items():
                precision_per_recall[str(recall)][key] = np.nan
    # print 'precision  %.2f' % np.mean(precision)
    return precision_per_recall, vis, ranks, exp_done ,[files_e,files_g]



def get_mAP(boxes_e, scores, boxes_g, iou_thresholds):
    # boxes are 1 x N x 9
    B, Ne, _ = list(boxes_e.shape)
    B, Ng, _ = list(boxes_g.shape)
    assert(B==1)
    boxes_e = np.reshape(boxes_e, (B*Ne, 9))
    boxes_g = np.reshape(boxes_g, (B*Ng, 9))
    corners_e = utils_geom.transform_boxes3D_to_corners_py(boxes_e)
    corners_g = utils_geom.transform_boxes3D_to_corners_py(boxes_g)
    # print("e", boxes_e, "g", boxes_g, "score", scores)
    scores = scores.flatten()
    # size [N, 8, 3]
    ious = np.zeros((Ne, Ng), dtype=np.float32)
    for i in list(range(Ne)):
        for j in list(range(Ng)):
            if(boxes_e[i,3]>0 and boxes_g[j,3]>0):
                iou_single, iou_2d_single = utils_box.box3d_iou(corners_e[i], corners_g[j])
                ious[i,j] = iou_single
    maps = []
    precisions_all = []
    for iou_threshold in iou_thresholds:
        map3d, precision, recall, overlaps= utils_ap.compute_ap(
            "box3D_"+str(iou_threshold), scores, ious, iou_threshold=iou_threshold)
        maps.append(map3d)
        precisions_all.append(precision[-2])

    maps = np.stack(maps, axis=0).astype(np.float32)
    precisions_all = np.stack(precisions_all, axis=0).astype(np.float32)

    if np.isnan(maps).any():
        print('got these nans in maps; setting to zero:', maps)
        maps[np.isnan(maps)] = 0.0
        # assert(False)
    
    # print("maps", maps)
    return maps,precisions_all,scores,ious



#@numba.jit
def get_thresholds(scores, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van',
                   'person_sitting', 'car', 'tractor', 'trailer']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in list(range(num_gt)):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in list(range(num_gt)):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in list(range(num_dt)):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in list(range(K)):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in list(range(N)):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    assert(False) # rotate_iou_gpu_eval (from the old nms_gpu) seems to require cudatoolkit=7.5, which seems unavailable
    # riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in list(range(N)):
        for j in list(range(K)):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / (EPS + ua)
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


#@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in list(range(det_size)):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in list(range(gt_size)):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in list(range(det_size)):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in list(range(det_size)):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in list(range(dc_bboxes.shape[0])):
                for j in list(range(det_size)):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in list(range(delta_idx)):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [remain_num]
    elif remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


#@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in list(range(gt_nums.shape[0])):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["location"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["location"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in list(range(num_part)):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = [np.zeros([f['location'].shape[0],5],
                              dtype=np.float64)
                     for f in gt_annos]
    dt_datas_list = [np.concatenate([np.zeros([f['location'].shape[0],5],
                                              dtype=np.float64),
                                     f['score'].reshape(-1,1)],
                                    axis=1)
                     for f in dt_annos]
    ignored_gts = [np.zeros(f['location'].shape[0], dtype=np.int64)
                   for f in gt_annos]
    ignored_dts = [np.zeros(f['location'].shape[0], dtype=np.int64)
                   for f in dt_annos]
    dontcares = [np.zeros([0,4], dtype=np.float64)
                 for f in gt_annos]
    total_dc_num = np.zeros(len(gt_annos), dtype=np.int64)
    total_num_valid_gt = sum(f['location'].shape[0]
                             for f in gt_annos)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dts,
            dontcares, total_dc_num, total_num_valid_gt)
    

# def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
#     gt_datas_list = []
#     dt_datas_list = []
#     total_dc_num = []
#     ignored_gts, ignored_dets, dontcares = [], [], []
#     total_num_valid_gt = 0
#     for i in list(range(len(gt_annos))):
#         rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
#         num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
#         ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
#         ignored_dets.append(np.array(ignored_det, dtype=np.int64))
#         if len(dc_bboxes) == 0:
#             dc_bboxes = np.zeros((0, 4)).astype(np.float64)
#         else:
#             dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
#         total_dc_num.append(dc_bboxes.shape[0])
#         dontcares.append(dc_bboxes)
#         total_num_valid_gt += num_valid_gt
#         gt_datas = np.concatenate(
#             [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
#         dt_datas = np.concatenate([
#             dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
#             dt_annos[i]["score"][..., np.newaxis]
#         ], 1)
#         gt_datas_list.append(gt_datas)
#         dt_datas_list.append(dt_datas)
#     total_dc_num = np.stack(total_dc_num, axis=0)
#     return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
#             total_dc_num, total_num_valid_gt)

def subsample_embs_2D(emb_e, emb_g, rgb, samps=10):
    # emb_e and _g are images
    B, H, W, C = emb_e.shape
    # TODO: validate the shape of emb_e, mainly the order of axis
    F = 10 # yields a 21x21 patch
    patches_e, patches_g, patches_r = [], [], []
    for e, g, r in zip(emb_e, emb_g, rgb):
        for samp in range(samps):
            row = np.random.randint(F, H-F)
            col = np.random.randint(F, W-F)
            patch_e = e[row-F:row+F+1,col-F:col+F+1]
            patch_g = g[row-F:row+F+1,col-F:col+F+1]
            patch_r = r[row-F:row+F+1,col-F:col+F+1]
            # print patch_e.shape
            patches_e.append(patch_e)
            patches_g.append(patch_g)
            patches_r.append(patch_r)
    # st()
    patches_e = np.stack(patches_e, axis=0)
    patches_g = np.stack(patches_g, axis=0)
    patches_r = np.stack(patches_r, axis=0)
    
    return patches_e, patches_g, patches_r
    # these are B*samp x F x F x C

def subsample_embs_3D(emb_e, emb_g, valid, rgb, samps=10):
    # emb_e and _g are voxel grids
    
    B, H, W, D, C = emb_e.shape
    # F = 2 # yields a 5x5x5 patch

    F = 1 # yields a 3x3x3 patch
    START =F
    ENDf = lambda x,y: x - y

    all_patches_e, all_patches_g, all_patches_r= [], [], []

    total_count = 0

    for e, g, v, r in zip(emb_e, emb_g, valid, rgb):
        patches_e, patches_g, patches_r = [], [], []
        count = 0
        # inds = np.where(np.not_equal(r[:,:,:,0], 127))
        # cond = np.not_equal(r[:,:,:,0], 127)
        # cond = np.logical_and(np.greater(v, 0.0), np.not_equal(r[:,:,:,0], 127))
        cond = np.greater(v, 0.0)
        # print cond.shape
        # print np.sum(cond)
        rows = list(range(START, ENDf(H,START)))
        cols = list(range(START, ENDf(W,START)))
        deps = list(range(START, ENDf(D,START)))
        random.shuffle(rows)
        random.shuffle(cols)
        random.shuffle(deps)
        for row in rows:
            for col in cols:
                for dep in deps:
                    if cond[row,col,dep]:
                        count += 1
                        patch_e = e[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_g = g[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_r = r[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patches_e.append(patch_e)
                        patches_g.append(patch_g)
                        patches_r.append(patch_r)
    
        # # print 'got %d 3D patches' % count
        # if count > samps:
        #     patches_e = patches_e[:samps]
        #     patches_g = patches_g[:samps]
        #     patches_r = patches_r[:samps]

        # grab some of these per ex in the batch
        for c in range(min(samps, count)):
            total_count += 1
            all_patches_e.append(patches_e[c])
            all_patches_g.append(patches_g[c])
            all_patches_r.append(patches_r[c])
                    
    if total_count > 0:
        all_patches_e = np.stack(all_patches_e, axis=0)
        all_patches_g = np.stack(all_patches_g, axis=0)
        all_patches_r = np.stack(all_patches_r, axis=0)
    
    return all_patches_e, all_patches_g, all_patches_r
    # these are B*samp x F x F x F x C


def subsample_embs_3D_o(emb_e, emb_g, valid, rgb, classes=None, samps=10):
    # emb_e and _g are voxel grids
    
    B, H, W, D, C = emb_e.shape
    # F = 2 # yields a 5x5x5 patch
    # take the entire box
    F = hyp.BOX_SIZE//2
    START = F
    ENDf = lambda x,y: y+1

    all_patches_e, all_patches_g, all_patches_r, all_classes = [], [], [], []

    total_count = 0

    for e, g, v, r, class_val in zip(emb_e, emb_g, valid, rgb, classes):
        patches_e, patches_g, patches_r,classes = [], [], [], []
        count = 0
        # inds = np.where(np.not_equal(r[:,:,:,0], 127))
        # cond = np.not_equal(r[:,:,:,0], 127)
        # cond = np.logical_and(np.greater(v, 0.0), np.not_equal(r[:,:,:,0], 127))
        cond = np.greater(v, 0.0)
        # print cond.shape
        # print np.sum(cond)
        rows = list(range(START, ENDf(H,START)))
        cols = list(range(START, ENDf(W,START)))
        deps = list(range(START, ENDf(D,START)))
        random.shuffle(rows)
        random.shuffle(cols)
        random.shuffle(deps)
        for row in rows:
            for col in cols:
                for dep in deps:
                    if cond[row,col,dep]:
                        count += 1
                        patch_e = e[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_g = g[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_r = r[row-F:row+F+1,col-F:col+F+1]
                        patches_e.append(patch_e)
                        patches_g.append(patch_g)
                        patches_r.append(patch_r)
                        classes.append(class_val)
    
        # # print 'got %d 3D patches' % count
        # if count > samps:
        #     patches_e = patches_e[:samps]
        #     patches_g = patches_g[:samps]
        #     patches_r = patches_r[:samps]

        # grab some of these per ex in the batch
        for c in range(min(samps, count)):
            total_count += 1
            all_patches_e.append(patches_e[c])
            all_patches_g.append(patches_g[c])
            all_patches_r.append(patches_r[c])
            all_classes.append(classes[c])
    if total_count > 0:
        all_patches_e = np.stack(all_patches_e, axis=0)
        all_patches_g = np.stack(all_patches_g, axis=0)
        all_patches_r = np.stack(all_patches_r, axis=0)
        all_classes = np.stack(all_classes, axis=0)
    return all_patches_e, all_patches_g, all_patches_r, all_classes

def subsample_embs_3D_o_cuda(emb_e, emb_g, valid, rgb, classes=None,filenames=None, samps=10):
    # emb_e and _g are voxel grids
    B, C, H, W, D = list(emb_e.shape)
    # F = 2 # yields a 5x5x5 patch
    # take the entire box
    F = hyp.BOX_SIZE//2
    START = F
    ENDf = lambda x,y: y+1

    all_patches_e, all_patches_g, all_patches_r, all_classes, all_filenames = [], [], [], [], []

    total_count = 0
    for e, g, v, r, class_val,file in zip(emb_e, emb_g, valid, rgb, classes,filenames):
        patches_e, patches_g, patches_r,classes,filenames = [], [], [], [],[]
        count = 0
        # inds = np.where(np.not_equal(r[:,:,:,0], 127))
        # cond = np.not_equal(r[:,:,:,0], 127)
        # cond = np.logical_and(np.greater(v, 0.0), np.not_equal(r[:,:,:,0], 127))
        cond = torch.gt(v, 0.0)
        # print cond.shape
        # print np.sum(cond)
        rows = list(range(START, ENDf(H,START)))
        cols = list(range(START, ENDf(W,START)))
        deps = list(range(START, ENDf(D,START)))
        random.shuffle(rows)
        random.shuffle(cols)
        random.shuffle(deps)
        for row in rows:
            for col in cols:
                for dep in deps:
                    if cond[:,row,col,dep]:
                        count += 1
                        patch_e = e[:,row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_g = g[:,row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_r = r[row-F:row+F+1,col-F:col+F+1,:]
                        patches_e.append(patch_e)
                        patches_g.append(patch_g)
                        patches_r.append(patch_r)
                        classes.append(class_val)
                        filenames.append(file)
        # grab some of these per ex in the batch
        for c in range(min(samps, count)):
            total_count += 1
            all_patches_e.append(patches_e[c])
            all_patches_g.append(patches_g[c])
            all_patches_r.append(patches_r[c])
            all_classes.append(classes[c])
            all_filenames.append(filenames[c])
    # st()   
    if total_count > 0:
        all_patches_e = torch.stack(all_patches_e, axis=0)
        all_patches_g = torch.stack(all_patches_g, axis=0)
        all_patches_r = np.stack(all_patches_r, axis=0)
        all_classes = np.stack(all_classes, axis=0)
        all_filenames = np.stack(all_filenames, axis=0)

    return all_patches_e, all_patches_g, all_patches_r, all_classes, all_filenames


'''

def eval_class_v3(gt_annos,
                  dt_annos,
                  current_classes,
                  difficultys,
                  metric,
                  min_overlaps,
                  compute_aos=False,
                  num_parts=50):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
                 patch_g = g[row-F:row+F+1,col-F:col+F+1]
            patch_r = r[row-F:row+F+1,col-F:col+F+1]
            # print patch_e.shape
            patches_e.append(patch_e)
            patches_g.append(patch_g)
            patches_r.append(patch_r)

    patches_e = np.stack(patches_e, axis=0)
    patches_g = np.stack(patches_g, axis=0)
    patches_r = np.stack(patches_r, axis=0)
    
    return patches_e, patches_g, patches_r
    # these are B*samp x F x F x C

def subsample_embs_3D(emb_e, emb_g, valid, rgb, samps=10):
    # emb_e and _g are voxel grids
    
    B, H, W, D, C = emb_e.shape

    # F = 2 # yields a 5x5x5 patch
    F = 1 # yields a 3x3x3 patch

    all_patches_e, all_patches_g, all_patches_r = [], [], []
    total_count = 0

    for e, g, v, r in zip(emb_e, emb_g, valid, rgb):
        patches_e, patches_g, patches_r = [], [], []
        count = 0
        
        # inds = np.where(np.not_equal(r[:,:,:,0], 127))
        # cond = np.not_equal(r[:,:,:,0], 127)
        # cond = np.logical_and(np.greater(v, 0.0), np.not_equal(r[:,:,:,0], 127))
        cond = np.greater(v, 0.0)
        # print cond.shape
        # print np.sum(cond)
        rows = list(range(F, H-F))
        cols = list(range(F, W-F))
        deps = list(range(F, D-F))
        random.shuffle(rows)
        random.shuffle(cols)
        random.shuffle(deps)
        for row in rows:
            for col in cols:
                for dep in deps:
                    if cond[row,col,dep]:
                        count += 1
                        patch_e = e[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_g = g[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patch_r = r[row-F:row+F+1,col-F:col+F+1,dep-F:dep+F+1]
                        patches_e.append(patch_e)
                        patches_g.append(patch_g)
                        patches_r.append(patch_r)
    
        # # print 'got %d 3D patches' % count
        # if count > samps:
        #     patches_e = patches_e[:samps]
        #     patches_g = patches_g[:samps]
        #     patches_r = patches_r[:samps]

        # grab some of these per ex in the batch
        for c in list(range(min(samps, count))):
            total_count += 1
            all_patches_e.append(patches_e[c])
            all_patches_g.append(patches_g[c])
            all_patches_r.append(patches_r[c])
                    
    if total_count > 0:
        all_patches_e = np.stack(all_patches_e, axis=0)
        all_patches_g = np.stack(all_patches_g, axis=0)
        all_patches_r = np.stack(all_patches_r, axis=0)
    
    return all_patches_e, all_patches_g, all_patches_r
    # these are B*samp x F x F x F x C

'''
