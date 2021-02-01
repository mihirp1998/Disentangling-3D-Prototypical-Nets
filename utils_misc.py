import random
import tensorboardX
import torch
import torch.nn as nn
import numpy as np
import utils_vox
import utils_improc
import utils_geom
import utils_basic
import utils_samp
import ipdb
st = ipdb.set_trace

def add_loss(name, total_loss, loss, coeff, summ_writer):
    # summ_writer should be Summ_writer object in utils_improc
    summ_writer.summ_scalar('unscaled_%s' % name, loss)
    summ_writer.summ_scalar('scaled_%s' % name, coeff*loss)

    total_loss = total_loss + coeff*loss
    return total_loss

# some code from: https://github.com/suruoxi/DistanceWeightedSampling
class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
        super(MarginLoss, self).__init__()
        self._margin = margin
        self._nu = nu

    def forward(self, anchors, positives, negatives, beta, a_indices=None):
        d_ap = torch.sqrt(torch.sum((positives - anchors)**2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors)**2, dim=1) + 1e-8)

        pos_loss = torch.clamp(d_ap - beta + self._margin, min=0.0)
        neg_loss = torch.clamp(beta - d_an + self._margin, min=0.0)

        pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))

        loss = torch.sum(pos_loss + neg_loss) / (1e-4 + pair_cnt)
        return loss, pair_cnt



class DistanceWeightedSampling(nn.Module):
    '''
    parameters
    ----------
    batch_k: int
        number of images per class

    Inputs:
        data: input tensor with shape (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx

    '''

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize=False, **kwargs):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize
        
    def get_distance(self, x):
        square = torch.sum(x**2, dim=1, keepdims=True)
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        return torch.sqrt(distance_square + torch.eye(x.shape[0], device=torch.device('cuda')))

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape

        debug = False
        # debug = True
        if debug:
            np.set_printoptions(precision=3, suppress=True)
            print(x[:,:5])
            print(x.shape)
        
        distance = self.get_distance(x)
        
        distance = torch.clamp(distance, min=self.cutoff)
        if debug:
            print('distance:')#, end=' ')
            print(distance.detach().cpu().numpy())

        log_weights = ((2.0 - float(d)) * torch.log(distance)
                       - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))

        if debug:
            print('log_weights:')#, end=' ')
            print(log_weights.detach().cpu().numpy())
        
        weights = torch.exp(log_weights - torch.max(log_weights))

        if debug:
            print('weights:')#, end=' ')
            print(weights.detach().cpu().numpy())

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones_like(weights)
        for i in range(0,n,k):
            mask[i:i+k, i:i+k] = 0
            
        if debug:
            print('mask:')#, end=' ')
            print(mask.detach().cpu().numpy())
            print('dist < nonzero:')#, end=' ')
            print((distance < self.nonzero_loss_cutoff).float().detach().cpu().numpy())

        # let's eliminate nans and zeros immediately
        weights[torch.isnan(weights)] = 1.0
        weights[weights < 1e-2] = 1e-2

        weights = weights * mask * (distance < self.nonzero_loss_cutoff).float()
        if debug:
            print('masked weights:')#, end=' ')
            print(weights.detach().cpu().numpy())
        
        weights = weights.detach().cpu().numpy()

        if debug:
            print('np weights:')#, end=' ')
            print(weights)
        
        # weights[np.isnan(weights)] = 1.0
        # weights[weights < 1e-2] = 1e-2

        if debug:
            print('clean weights:')#, end=' ')
            print(weights)

        # careful divison here
        weights = weights / (1e-4 + np.sum(weights, axis=1, keepdims=True))
            
        if debug:
            print('new weights:')#, end=' ')
            # print(weights.detach().cpu().numpy())
            print(weights)
        a_indices = []
        p_indices = []
        n_indices = []
        # np_weights = weights.cpu().detach().numpy()
        np_weights = weights
        for i in range(n):
            block_idx = i // k
            try:
                n_indices += np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()
            for j in range(block_idx * k, (block_idx + 1)*k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)
        return a_indices, x[a_indices], x[p_indices], x[n_indices], x

def shuffle_valid_and_sink_invalid_boxes(boxes, tids, scores):
    # put the good boxes shuffled at the top;
    # sink the bad boxes to the bottom.

    # boxes are B x N x D
    # tids are B x N
    # scores are B x N
    B, N, D = list(boxes.shape)

    boxes_new = torch.zeros_like(boxes)
    tids_new = -1*torch.ones_like(tids)
    scores_new = torch.zeros_like(scores)

    for b in list(range(B)):

        # for the sake of training,
        # we want to mix up the ordering
        index_shuf = list(range(N))
        np.random.shuffle(index_shuf)

        count = 0
        for i in list(range(N)):
            j = index_shuf[i]
            box = boxes[b,j]
            tid = tids[b,j]
            score = scores[b,j]
            if score > 0.0:
                boxes_new[b,count] = box
                tids_new[b,count] = tid
                scores_new[b,count] = score
                count += 1

    return boxes_new, tids_new, scores_new

def get_target_scored_box_single(target, boxes, tids, scores):
    # boxes are N x D
    # tids are N and int32
    # scores are N
    # here we retrieve one target box
    N, D = list(boxes.shape)
    
    box_ = torch.ones(D)
    score_ = torch.zeros(1)
    # print 'target = %d' % (target),

    count = 0
    for i in range(N):
        box = boxes[i]
        tid = tids[i]
        score = scores[i]
        # print 'target = %d; tid = %d; score = %.2f' % (target, tid, score)
        if score > 0.0 and tid==target:
            # print 'got it:',
            # print box,
            # print score
            return box, score
    # did not find it; return empty stuff (with score 0)
    return box_, score_

def get_target_traj(targets, boxlist_s, tidlist_s, scorelist_s):
    # targets are B
    # boxlist_s are B x S x N x D
    # tidlist_s are B x S x N
    # scorelist_s are B x S x N
    
    B, S, N, D = list(boxlist_s.shape)
    # (no asserts on shape; boxlist could instead be lrtlist)

    # return box_traj for the target, sized B x S x D
    # and also the score_traj, sized B x S
    # (note the object may not live across all frames)

    box_traj = torch.zeros(B, S, D)
    score_traj = torch.zeros(B, S)
    for b in range(B):
        for s in range(S):
            box_, score_ = get_target_scored_box_single(targets[b], boxlist_s[b,s], tidlist_s[b,s], scorelist_s[b,s])
            box_traj[b,s] = box_
            score_traj[b,s] = score_
    return box_traj.cuda(), score_traj.cuda()

def collect_object_info(lrtlist_camRs, tidlist_s, scorelist_s, pix_T_cams, K, mod='', do_vis=True, summ_writer=None):
    # rgbRs, xyz_camRs, 
    # rgbRs is B x S x H x W x 3
    # xyz_camRs is B x S x V x 3
    # lrtlist_camRs is B x S x N x 19
    # tidlist_s is B x S x N
    # scorelist_s is B x S x N
    # pix_T_cams is B x S x 4 x 4
    
    # K (int): number of objects to collect
    B, S, N, D = list(lrtlist_camRs.shape)
    

    # this returns a bunch of tensors that begin with dim K
    # these tensors are object-centric: along S is all the info for that particular obj
    # this is in contrast to something like boxes, which is frame-centric
    
    obj_score_traj = []
    obj_lrt_traj = []
    obj_occ_traj = []
    obj_unp_traj = []
    for target_ind in range(K):
        target_tid = tidlist_s[:,0,target_ind]
        tid_traj = torch.reshape(target_tid, [B, 1]).repeat(1, S)

        # extract its traj from the full tensors
        lrt_traj, score_traj = get_target_traj(target_tid, lrtlist_camRs, tidlist_s, scorelist_s)
        # lrt_traj is B x S x 19
        # score_traj is B x S

        obj_lrt_traj.append(lrt_traj)
        obj_score_traj.append(score_traj)

        # # voxelize it
        # occ_traj = []
        # unp_traj = []
        # for s in range(S):
        #     box_ = box_traj[:,s]
        #     xyz_list = tf.unstack(xyz_camRs[:,s], axis=0)
            
        #     occ = utils_vox.voxelize_obj_using_xyz_list(xyz_list, box_, protos[:,s])
        #     # occ is B x ZH x ZW x ZD
        #     occ_traj.append(occ)
            
        #     unp = utils_vox.unproject_image_to_zoom(rgbRs[:,s], box_, ZH, ZW, ZD, pix_T_cams[:,s])
        #     # unp is B x ZH x ZW x ZD x 3
        #     unp_traj.append(unp)
            
        # occ_traj = tf.stack(occ_traj, axis=1)
        # obj_occ_traj.append(occ_traj)
        
        # unp_traj = tf.stack(unp_traj, axis=1)
        # obj_unp_traj.append(unp_traj)

        ## this works, if rgbRs is provided
        # if target_ind==0 and do_vis and (summ_writer is not None):
        #     summ_writer.summ_lrtlist('target_lrt_traj_g',
        #                               rgbRs[:,0],
        #                               lrt_traj, # note S will be treated as the N dim
        #                               score_traj,
        #                               tid_traj,
        #                               pix_T_cams[:,0])
        #     # print_shape(unp_traj[0])
        #     # print_shape(occ_traj[0])
        #     # utils_improc.summ_unps('target_obj_unp_traj', tf.unstack(unp_traj, axis=1), tf.unstack(occ_traj, axis=1))
        #     # # utils_improc.summ_occs('target_obj_occ_traj', tf.unstack(occ_traj, axis=1))

    ## stack up
    obj_lrt_traj = torch.stack(obj_lrt_traj, axis=0)
    # this is K x B x S x 7
    obj_score_traj = torch.stack(obj_score_traj, axis=0)
    # # this is K x B x S
    # obj_occ_traj = tf.stack(obj_occ_traj, axis=0)
    # # this is K x B x ZH x ZW x ZD
    # obj_unp_traj = tf.stack(obj_unp_traj, axis=0)
    # # this is K x B x ZH x ZW x ZD x 3

    # return obj_lrt_traj, obj_score_traj, obj_occ_traj, obj_unp_traj
    return obj_lrt_traj, obj_score_traj#, obj_occ_traj, obj_unp_traj

def rescore_boxlist_with_inbound(boxlist, tidlist, Z, Y, X):
    # boxlist is B x N x 9
    B, N, D = list(boxlist.shape)
    assert(D==9)
    xyzlist = boxlist[:,:,:3]
    # this is B x N x 3
    # a box at 0,0,0 is probably invalid
    EPS = 1e-6
    # nonzerolist = (torch.sum(torch.abs(xyzlist), dim=2) > EPS).float()
    validlist = 1.0-(torch.eq(tidlist, -1*torch.ones_like(tidlist))).float()
    # this is B x N
    inboundlist = utils_vox.get_inbounds(xyzlist, Z, Y, X, already_mem=False).float()
    scorelist = validlist * inboundlist
    return scorelist

def get_gt_flow(obj_lrtlist_camRs,
                obj_scorelist,
                camRs_T_camXs,
                Z, Y, X, 
                K=2,
                mod='',
                vis=True,
                summ_writer=None):
    # this constructs the flow field according to the given
    # box trajectories (obj_lrtlist_camRs) (collected from a moving camR)
    # and egomotion (encoded in camRs_T_camXs)
    # (so they do not take into account egomotion)
    # so, we first generate the flow for all the objects,
    # then in the background, put the ego flow
    
    N, B, S, D = list(obj_lrtlist_camRs.shape)
    assert(S==2) # as a flow util, this expects S=2

    flows = []
    masks = []
    for k in range(K):
        obj_masklistR0 = utils_vox.assemble_padded_obj_masklist(
            obj_lrtlist_camRs[k,:,0:1],
            obj_scorelist[k,:,0:1],
            Z, Y, X,
            coeff=1.0)
        # this is B x 1(N) x 1(C) x Z x Y x Z
        # obj_masklistR0 = obj_masklistR0.squeeze(1)
        # this is B x 1 x Z x Y x X
        obj_mask0 = obj_masklistR0.squeeze(1)
        # this is B x 1 x Z x Y x X

        camR_T_cam0 = camRs_T_camXs[:,0]
        camR_T_cam1 = camRs_T_camXs[:,1]
        cam0_T_camR = utils_geom.safe_inverse(camR_T_cam0)
        cam1_T_camR = utils_geom.safe_inverse(camR_T_cam1)
        # camR0_T_camR1 = camR0_T_camRs[:,1]
        # camR1_T_camR0 = utils_geom.safe_inverse(camR0_T_camR1)

        # obj_masklistA1 = utils_vox.apply_4x4_to_vox(camR1_T_camR0, obj_masklistA0)
        # if vis and (summ_writer is not None):
        #     summ_writer.summ_occ('flow/obj%d_maskA0' % k, obj_masklistA0)
        #     summ_writer.summ_occ('flow/obj%d_maskA1' % k, obj_masklistA1)

        if vis and (summ_writer is not None):
            # summ_writer.summ_occ('flow/obj%d_mask0' % k, obj_mask0)
            summ_writer.summ_oned('flow/obj%d_mask0' % k, torch.mean(obj_mask0, 3))
        
        _, ref_T_objs_list = utils_geom.split_lrtlist(obj_lrtlist_camRs[k])
        # this is B x S x 4 x 4
        ref_T_obj0 = ref_T_objs_list[:,0]
        ref_T_obj1 = ref_T_objs_list[:,1]
        obj0_T_ref = utils_geom.safe_inverse(ref_T_obj0)
        obj1_T_ref = utils_geom.safe_inverse(ref_T_obj1)
        # these are B x 4 x 4
        
        mem_T_ref = utils_vox.get_mem_T_ref(B, Z, Y, X)
        ref_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)

        ref1_T_ref0 = utils_basic.matmul2(ref_T_obj1, obj0_T_ref)
        cam1_T_cam0 = utils_basic.matmul3(cam1_T_camR, ref1_T_ref0, camR_T_cam0)
        mem1_T_mem0 = utils_basic.matmul3(mem_T_ref, cam1_T_cam0, ref_T_mem)

        xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
        xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)

        xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
        xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)

        # only use these displaced points within the obj mask
        # obj_mask03 = obj_mask0.view(B, Z, Y, X, 1).repeat(1, 1, 1, 1, 3)
        obj_mask0 = obj_mask0.view(B, Z, Y, X, 1)
        # # xyz_mem1[(obj_mask03 < 1.0).bool()] = xyz_mem0
        # cond = (obj_mask03 < 1.0).float()
        cond = (obj_mask0 > 0.0).float()
        xyz_mem1 = cond*xyz_mem1 + (1.0-cond)*xyz_mem0

        flow = xyz_mem1 - xyz_mem0
        flow = flow.permute(0, 4, 1, 2, 3)
        obj_mask0 = obj_mask0.permute(0, 4, 1, 2, 3)

        # if vis and k==0:
        if vis:
            summ_writer.summ_3D_flow('flow/gt_%d' % k, flow, clip=4.0)

        masks.append(obj_mask0)
        flows.append(flow)

    camR_T_cam0 = camRs_T_camXs[:,0]
    camR_T_cam1 = camRs_T_camXs[:,1]
    cam0_T_camR = utils_geom.safe_inverse(camR_T_cam0)
    cam1_T_camR = utils_geom.safe_inverse(camR_T_cam1)

    mem_T_ref = utils_vox.get_mem_T_ref(B, Z, Y, X)
    ref_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)

    cam1_T_cam0 = utils_basic.matmul2(cam1_T_camR, camR_T_cam0)
    mem1_T_mem0 = utils_basic.matmul3(mem_T_ref, cam1_T_cam0, ref_T_mem)

    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
    xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)

    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)

    flow = xyz_mem1 - xyz_mem0
    flow = flow.permute(0, 4, 1, 2, 3)

    bkg_flow = flow

    # allow zero motion in the bkg
    any_mask = torch.max(torch.stack(masks, axis=0), axis=0)[0]
    masks.append(1.0-any_mask)
    flows.append(bkg_flow)

    flows = torch.stack(flows, axis=0)
    masks = torch.stack(masks, axis=0)
    masks = masks.repeat(1, 1, 3, 1, 1, 1)
    flow = utils_basic.reduce_masked_mean(flows, masks, dim=0)

    if vis:
        summ_writer.summ_3D_flow('flow/gt_complete', flow, clip=4.0)

    # flow is shaped B x 3 x D x H x W
    return flow

def get_synth_flow(occs,
                   unps,
                   summ_writer,
                   sometimes_zero=False,
                   do_vis=False):
    B,S,C,Z,Y,X = list(occs.shape)
    assert(S==2,C==1)

    # we do not sample any rotations here, to keep the distribution purely
    # uniform across all translations
    # (rotation ruins this, since the pivot point is at the camera)
    cam1_T_cam0 = [utils_geom.get_random_rt(B, r_amount=0.0, t_amount=1.0), # large motion
                   utils_geom.get_random_rt(B, r_amount=0.0, t_amount=0.1, # small motion
                                            sometimes_zero=sometimes_zero)]
    cam1_T_cam0 = random.sample(cam1_T_cam0, k=1)[0]

    occ0 = occs[:,0]
    unp0 = unps[:,0]
    occ1 = utils_vox.apply_4x4_to_vox(cam1_T_cam0, occ0, binary_feat=True)
    unp1 = utils_vox.apply_4x4_to_vox(cam1_T_cam0, unp0)
    occs = [occ0, occ1]
    unps = [unp0, unp1]

    if do_vis:
        summ_writer.summ_occs('synth/occs', occs)
        summ_writer.summ_unps('synth/unps', unps, occs)
        
    mem_T_cam = utils_vox.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)
    mem1_T_mem0 = utils_basic.matmul3(mem_T_cam, cam1_T_cam0, cam_T_mem)
    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
    xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)
    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)
    flow = xyz_mem1-xyz_mem0
    # this is B x Z x Y x X x 3
    flow = flow.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    if do_vis:
        summ_writer.summ_3D_flow('synth/flow', flow, clip=2.0)

    if do_vis:
        occ0_e = utils_samp.backwarp_using_3D_flow(occ1, flow, binary_feat=True)
        unp0_e = utils_samp.backwarp_using_3D_flow(unp1, flow)
        summ_writer.summ_occs('synth/occs_stab', [occ0, occ0_e])
        summ_writer.summ_unps('synth/unps_stab', [unp0, unp0_e], [occ0, occ0_e])

    occs = torch.stack(occs, dim=1)
    unps = torch.stack(unps, dim=1)

    return occs, unps, flow, cam1_T_cam0

def get_safe_samples(valid, dims, N_to_sample, mode='3D', tol=5.0):
    N, C = list(valid.shape)
    assert(C==1)
    assert(N==np.prod(dims))
    inds, locs, valids = get_safe_samples_py(valid, dims, N_to_sample, mode=mode, tol=tol)
    inds = torch.from_numpy(inds).to('cuda')
    locs = torch.from_numpy(locs).to('cuda')
    valids = torch.from_numpy(valids).to('cuda')
    
    inds = torch.reshape(inds, [N_to_sample, 1])
    inds = inds.long()
    if mode=='3D':
        locs = torch.reshape(locs, [N_to_sample, 3])
    elif mode=='2D':
        locs = torch.reshape(locs, [N_to_sample, 2])
    else:
        assert(False)# choose 3D or 2D please
    locs = locs.float()
    valids = torch.reshape(valids, [N_to_sample])
    valids = valids.float()
    return inds, locs, valids

def get_safe_samples_py(valid, dims, N_to_sample, mode='3D', tol=5.0):
    if mode=='3D':
        Z, Y, X = dims
    elif mode=='2D':
        Y, X = dims
    else:
        assert(False) # please choose 2D or 3D
    valid = valid.detach().cpu()
    valid = np.reshape(valid, [-1])
    N_total = len(valid)
    # assert(N_to_sample < N_total) # otw we need a padding step, and maybe a mask in the loss
    initial_tol = tol

    all_inds = np.arange(N_total)
    # reshape instead of squeeze, in case one or zero come
    valid_inds = all_inds[np.reshape((np.where(valid > 0)), [-1])]
    N_valid = len(valid_inds)
    # print('initial tol = %.2f' % tol)
    # print('N_valid = %d' % N_valid)
    # print('N_to_sample = %d' % N_to_sample)
    if N_to_sample < N_valid:
        # ok we can proceed

        if mode=='3D':
            xyz = utils_basic.gridcloud3D_py(Z, Y, X)
            locs = xyz[np.reshape((np.where(valid > 0)), [-1])]
        elif mode=='2D':
            xy = utils_basic.gridcloud2D_py(Y, X)
            locs = xy[np.reshape((np.where(valid > 0)), [-1])]

        samples_ok = False
        nTries = 0
        while (not samples_ok):
            # print('sample try %d...' % nTries)
            nTries += 1
            sample_inds = np.random.permutation(N_valid).astype(np.int32)[:N_to_sample]
            samples_try = valid_inds[sample_inds]
            locs_try = locs[sample_inds]
            nn_dists = np.zeros([N_to_sample], np.float32)
            samples_ok = True # ok this might work

            for i, loc in enumerate(locs_try):
                # exclude the current samp
                other_locs0 = locs_try[:i]
                other_locs1 = locs_try[i+1:]
                other_locs = np.concatenate([other_locs0, other_locs1], axis=0) 
                dists = np.linalg.norm(
                    np.expand_dims(loc, axis=0).astype(np.float32) - other_locs.astype(np.float32), axis=1)
                mindist = np.min(dists)
                nn_dists[i] = mindist
                if mindist < tol:
                    samples_ok = False
            # ensure we do not get stuck here: every 100 tries, subtract 1px to make it easier
            tol = tol - nTries*0.01
        # print(locs_try)
        if tol < (initial_tol/2.0):
            print('warning: initial_tol = %.2f; final_tol = %.2f' % (initial_tol, tol))
        # utils_basic.print_stats_py('nn_dists_%s' % mode, nn_dists)

        # print('these look ok:')
        # print(samples_try[:10])
        valid = np.ones(N_to_sample, np.float32)
    else:
        print('not enough valid samples! returning a few fakes')
        if mode=='3D':
            perm = np.random.permutation(Z*Y*X)
        elif mode=='2D':
            perm = np.random.permutation(Y*X)
        else:
            assert(False) # 2D or 3D please
        samples_try = perm[:N_to_sample].astype(np.int32)
        # not enough valid samples, so we need some fake returns
        locs_try = np.zeros((N_to_sample, 3), np.float32)
        valid = np.zeros(N_to_sample, np.float32)
    return samples_try, locs_try, valid

def get_synth_flow_v2(xyz_cam0,
                      occ0,
                      unp0,
                      summ_writer,
                      sometimes_zero=False,
                      do_vis=False):
    # this version re-voxlizes occ1, rather than warp
    B,C,Z,Y,X = list(unp0.shape)
    assert(C==3)
    
    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)

    # we do not sample any rotations here, to keep the distribution purely
    # uniform across all translations
    # (rotation ruins this, since the pivot point is at the camera)
    cam1_T_cam0 = [utils_geom.get_random_rt(B, r_amount=0.0, t_amount=3.0), # large motion
                   utils_geom.get_random_rt(B, r_amount=0.0, t_amount=0.1, # small motion
                                            sometimes_zero=sometimes_zero)]
    cam1_T_cam0 = random.sample(cam1_T_cam0, k=1)[0]

    xyz_cam1 = utils_geom.apply_4x4(cam1_T_cam0, xyz_cam0)
    occ1 = utils_vox.voxelize_xyz(xyz_cam1, Z, Y, X)
    unp1 = utils_vox.apply_4x4_to_vox(cam1_T_cam0, unp0)
    occs = [occ0, occ1]
    unps = [unp0, unp1]

    if do_vis:
        summ_writer.summ_occs('synth/occs', occs)
        summ_writer.summ_unps('synth/unps', unps, occs)
        
    mem_T_cam = utils_vox.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)
    mem1_T_mem0 = utils_basic.matmul3(mem_T_cam, cam1_T_cam0, cam_T_mem)
    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
    xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)
    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)
    flow = xyz_mem1-xyz_mem0
    # this is B x Z x Y x X x 3
    flow = flow.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    if do_vis:
        summ_writer.summ_3D_flow('synth/flow', flow, clip=2.0)

    if do_vis:
        occ0_e = utils_samp.backwarp_using_3D_flow(occ1, flow, binary_feat=True)
        unp0_e = utils_samp.backwarp_using_3D_flow(unp1, flow)
        summ_writer.summ_occs('synth/occs_stab', [occ0, occ0_e])
        summ_writer.summ_unps('synth/unps_stab', [unp0, unp0_e], [occ0, occ0_e])

    occs = torch.stack(occs, dim=1)
    unps = torch.stack(unps, dim=1)

    return occs, unps, flow, cam1_T_cam0
