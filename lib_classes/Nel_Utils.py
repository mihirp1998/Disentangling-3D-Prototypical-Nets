import ipdb
st = ipdb.set_trace
import random
# st()
'''
pip install open3d-python==0.5.0.0
use this to install open3d. Other versions may give 
error when importing. Refer this issue:
https://github.com/pytorch/pytorch/issues/19739
'''
# make sure it is called before importing torch
#import open3d as o3d

import numpy as np
# import tensorflow as tf

import cv2
from sklearn.cluster import DBSCAN as skdbscan

# from cc3d import connected_components
import utils_improc
import utils_geom
import utils_basic
import utils_vox
import copy
import torch
import torch.nn.functional as F
import cross_corr
import hyperparams as hyp
EPS = 1e-6
dict_val = {'green_apple', 'small_banana', 'black_grapes', 'strawberry', 'big_banana', 'green_grapes', 'pear', 'tomato', 'red_grapes', 'red_peach', 'appricot', 'yellow_lemon', 'green_lemon', 'avocado', 'red_apple'}

def get_boxes_from_occ(occ, rgb, coord_mem):
    B, H, W, D, C = occ.shape

    # flow_mag = tf.reduce_mean(l2_on_axis(flow, axis=4), axis=3)
    # vis = utils_improc.oned2inferno(flow_mag)
    N = 10
    vis = utils_improc.back2color(rgb)
    # Adds a new dimension at the very start.
    coord_mem_exp = coord_mem.unsqueeze(0)
    coord_mem = coord_mem_exp.repeat(B, 1)

    # vis, boxes3D, scores, conn = tf.map_fn(get_boxes_from_occ_single, (occ, vis, coord_mem), dtype=(
    #     tf.uint8, tf.float32, tf.float32, tf.float32))

    vis_list = []
    boxes3D_list = []
    scores_list = []
    conn_list = []
    for occ_i, vis_i, coord_mem_i in zip(occ, vis, coord_mem):
        output_occ_single = get_boxes_from_occ_single((occ_i, vis_i, coord_mem_i))
        
        vis_list.append(output_occ_single[0])
        boxes3D_list.append(output_occ_single[1])
        scores_list.append(output_occ_single[2])
        conn_list.append(output_occ_single[3])

    vis = torch.stack(vis_list)
    boxes3D = torch.stack(boxes3D_list)
    scores = torch.stack(scores_list)
    conn = torch.stack(conn_list)

    
    boxes3D = boxes3D.view([B, N, 9])
    scores = scores.view([B, N])
    conn = conn.view([B, N, H, W, D, 1])
    # tf.summary.histogram('boxes3D', boxes3D)

    # xc, yc, zc, h, w, l, r = tf.unstack(boxes3D, axis=2)
    # utils_improc.draw_boxes2D_on_image(rgb, boxes, scores)

    
    # vis = utils_improc.preprocess_color(vis)
    # utils_improc.summ_rgb('boxes_vis', vis, bird=True)
    
    return vis, boxes3D, scores, conn    

def generate_shape():
    if hyp.dataset_name == "clevr":
        x = random.randint(7,15)
        z = random.randint(7,15)
        y = random.randint(7,12)
    else:
        st()
    return [z,y,x]
def get_boxes_from_occ_single(single_occ_info):
    # flow is H x W x D x 3
    (occ, vis,coord_mem) = single_occ_info
    if type(occ).__module__ == np.__name__:
        occ = torch.from_numpy(occ)
    occ_mag = utils_basic.l2_on_axis(occ, axis=3)
    # flow_mag is H x W x D x 1
    # vis = utils_improc.oned2inferno(flow_mag)
    # boxes3D, scores = tf.py_function(get_boxes_from_flow_mag_py, [flow_mag], (tf.float32, tf.float32))
    occ_mag_np = occ_mag.numpy()
    vis_np = vis.numpy()
    coord_mem_np = coord_mem.numpy()
    vis, boxes3D, scores, conn = get_boxes_from_occ_mag_py(occ_mag_np, vis_np,coord_mem_np)
    # tf.summary.histogram('conn_comps', conn_comps)
    # take a linspace of threhsolds between the min and max
    # for each thresh
    #   create a binary map
    #   turn this into labels with connected_components
    # vis all these
    # boxpoints =
    vis = torch.from_numpy(vis.astype(int))
    boxes3D = torch.from_numpy(boxes3D.astype(float))
    scores = torch.from_numpy(scores.astype(float))
    conn = torch.from_numpy(conn.astype(float))
    return vis, boxes3D, scores, conn




def make_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # if the dim is greater than 3 I expect the color
    if pts.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
            if pts[:, 3:].max() > 1. else pts[:, 3:])
    return pcd

def get_bounding_box_coordinates(merged_pts):
    """Merges all the pcds computes the bbox and returns it
    """
    xmax, xmin = np.max(merged_pts[:, 0], axis=0),\
        np.min(merged_pts[:, 0], axis=0)

    ymax, ymin = np.max(merged_pts[:, 1], axis=0),\
        np.min(merged_pts[:, 1], axis=0)

    zmax, zmin = np.max(merged_pts[:, 2], axis=0),\
        np.min(merged_pts[:, 2], axis=0)

    return np.asarray([xmin, xmax, ymin, ymax, zmin, zmax])

def form_eight_points_of_bbox(bbox_coords):
    xmin, ymin, zmin = bbox_coords[0:6:2]
    xmax, ymax, zmax = bbox_coords[1:6:2]
    eight_points = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],\
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]]
    return eight_points

def create_binary_mask(boxes,shape):
    canvas = torch.zeros(shape)
    for box in boxes:
        lower,upper = torch.unbind(box)
        xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
        xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]    
        canvas[zmin:zmax,ymin:ymax,xmin:xmax] = 1
    return canvas


def get_single_bbox_coords(merged_pts, vis=False):
    # merged_pts, merged_colors = merge_pcds(subt_pcd)
    # merged_pts, merged_colors = np.asarray(subt_pcd.points), np.asarray(subt_pcd.colors)
    bbox_coords = get_bounding_box_coordinates(merged_pts)
    if vis:
        # form the merged_pcd for visualization
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(merged_pts)
        # combined_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        # points for linesets
        points = form_eight_points_of_bbox(bbox_coords)

        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set, bbox_coords
        # print('visualize the scene with the bounding box')
        # o3d.visualization.draw_geometries([combined_pcd, line_set])
    else:
        return None, bbox_coords

# CLEVR_new_000599. small objects included

def cluster_using_dbscan(pcd_points,MAX_OBJECTS_IN_SCENE, vis=False):
    # pcd_points = pcd_points.numpy().astype(np.float32) 
    # pcd_points = pcd_points/40.0
    pcd = make_pcd(pcd_points)
    clustering = skdbscan(eps=1, min_samples=1).fit(pcd_points)
    clustered_pcd_indices = {}
    print("Number of objects found:", 1+np.max(clustering.labels_))
    for index, predicted_cluster in enumerate(clustering.labels_):
        if predicted_cluster not in clustered_pcd_indices:
            clustered_pcd_indices[predicted_cluster] = {'points': list()}
        clustered_pcd_indices[predicted_cluster]['points'].append(pcd_points[index])        
    
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=10.0, origin=[0, 0, 0])

    final_list = [pcd]
    bbox_list = []
    for predicted_cluster in clustered_pcd_indices:
        ret, bbox_coords = get_single_bbox_coords(np.asarray(clustered_pcd_indices[predicted_cluster]['points']), vis)
        xmin,xmax,ymin,ymax,zmin,zmax = bbox_coords
        final_list.append(ret)
        bbox_list.append(np.asarray([xmin,ymin,zmin,xmax,ymax,zmax]))
    # if vis:
    #     print("Visualize bounding boxes from features")
    #     o3d.visualization.draw_geometries(final_list)
    bbox_array = np.stack(bbox_list)
    padded_bbox_array = np.zeros((MAX_OBJECTS_IN_SCENE, 6))
    #Multiply x and y coords by -1 to bring them to discovery/GoennTf2's coordinate system
    padded_bbox_array[:, :4] *= -1
    padded_bbox_array[: bbox_array.shape[0]] = bbox_array
    return padded_bbox_array,pcd


def only_visualize(pcd,boxes):
    # pcd_points = pcd_points.numpy().astype(np.float32) 
    # pcd_points = pcd_points/40.0
    final_list = [pcd]
    bbox_list = []
    for bbox_coords in boxes[0]:
        xmin, ymin, zmin = bbox_coords[0:3]
        xmax, ymax, zmax = bbox_coords[3:6]
        points = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],\
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]]

        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

        colors = [[1, 0, 0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)    
        final_list.append(line_set)
    o3d.visualization.draw_geometries(final_list)


def yxz2xyz(box):
    B,N,_,_ = box.shape

    y = box[:,:,:,0]
    x = box[:,:,:,1]
    z = box[:,:,:,2]

    box = torch.stack([x,y,z],dim=-1)
    box = box.view([B,N,6])
    return box

def yxz2xyz_v2(box):
    B,N,_ = box.shape

    y = box[:,:,0]
    x = box[:,:,1]
    z = box[:,:,2]

    box = torch.stack([x,y,z],dim=-1)
    box = box.view([B,N,3])
    return box

def postproccess_for_scores(boxes_temp):
    return_boxes = []
    boxes = copy.deepcopy(boxes_temp)
    index = 1
    for i in boxes:
        i[:,:,index] = i[:,:,index] - 0.2
        i[:,:,index+3] = i[:,:,index+3] - 0.2
        return_boxes.append(i)
    return return_boxes

def postproccess_for_ap(boxes_temp):
    boxes = copy.deepcopy(boxes_temp)
    boxes = boxes.numpy()
    index = 1
    boxes[:,:,index] = boxes[:,:,index] + 0.2
    boxes[:,:,index+3] = boxes[:,:,index+3] + 0.2
    return boxes    

def meshgrid3D_py(H, W, D):
    x = np.linspace(0, H-1, H)
    y = np.linspace(0, W-1, W)
    z = np.linspace(0, D-1, D)
    xv, yv, zv = np.meshgrid(x, y, z)
    return xv, yv, zv

def get_boxes_from_occ_mag_py(occ_mag, image, coord_mem):
    # flow_mag is B x H x W x D x 1 and np.float32
    H, W, D, _ = occ_mag.shape
    # st()
    N = 10
    # print 'shape of flow_mag: %d x %d x %d' % (H, W, D)
    # np.save('flow_mag.npy', flow_mag)

    # adjust for numerical errors
    occ_mag = occ_mag*100.0
    # boxes2D = np.zeros([N, 4], dtype=np.float32)
    boxes3D = np.zeros([N, 9], dtype=np.float32)
    scores = np.zeros([N], dtype=np.float32)
    conn = np.zeros([N, H, W, D], dtype=np.float32)
    boxcount = 0

    mag = np.reshape(occ_mag, [H, W, D])
    mag_min, mag_max = np.min(mag), np.max(mag)
    # print('min, max = %.6f, %.6f' % (mag_min, mag_max))
    
    threshs = np.linspace(mag_min, mag_max, num=12)
    threshs = threshs[1:-1]
    # print('threshs:')
    # print(threshs)
    
    xg, yg, zg = meshgrid3D_py(W,H,D)
    box3D_list = []
    number = 0
    numberContour = 0
    uniqueBoxes = 0
    for ti, thresh in enumerate(threshs):
        # print 'working on thresh %d: %.2f' % (ti, thresh)
        mask = (mag > thresh).astype(np.int32)
        if np.sum(mask) > 8: # if we have a few pixels to connect up 
            labels = connected_components(mask)
            segids = [ x for x in np.unique(labels) if x != 0 ]
            for si, segid in enumerate(segids):
                extracted_vox = (labels == segid)
                if np.sum(extracted_vox) > 8: # if we have a few pixels to box up 
                    number +=1
                    # print 'segid = %d' % segid
                    # print extracted_vox.shape
                    y = yg[extracted_vox==1]
                    x = xg[extracted_vox==1]
                    z = zg[extracted_vox==1]

                    # find the oriented box in birdview
                    im = np.sum(extracted_vox, axis=0)
                    im = im.astype(np.uint8)
                    # print im.shape
                    contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        numberContour +=1
                        cnt = contours[0]
                        rect = cv2.minAreaRect(cnt)

                        # i want to clip at the index where YMAX dips under the ground
                        # and where YMIN reaches above some reasonable height

                        shift = coord_mem[2] #ymin
                        scale = float(H)/np.abs(float(coord_mem[3]-coord_mem[2])) #ymax -ymin
                        # st()
                        ymin_ = (coord_mem[6]-shift)*scale #floor
                        ymax_ = (coord_mem[7]-shift)*scale  #ceil

                        if ymin_ > ymax_:
                            # this is true if y points downards
                            ymax_, ymin_ = ymin_, ymax_
                            
                        ymin = np.clip(np.min(y), ymin_, ymax_)
                        ymax = np.clip(np.max(y), ymin_, ymax_)
                        # st()
                            
                        hei = ymax-ymin
                        yc = (ymax+ymin)/2.0

                        (zc,xc),(dep,wid),theta = rect
                        
                        box = cv2.boxPoints(rect)
                        if dep < wid:
                            # dep goes along the long side of an oriented car
                            theta += 90.0
                            wid, dep = dep, wid
                        theta = utils_geom.deg2rad(theta)

                        # boxes3D1 = tf.stack([x, y, z3d,h3d,w3d,l3d,ry], axis=2)
                        if boxcount < N:
                            # bx, by = np.split(box, axis=1)
                            # boxpoints[boxcount,:] = box

                            box3D = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
                            box3D = np.array(box3D).astype(np.float32)

                            already_have = False
                            for box3D_ in box3D_list:
                                if np.all(box3D_==box3D):
                                    already_have = True
                                    # print("we already have this")
                                    
                            # if (not already_have) and (wid > 0) and (hei > 0) and (dep > 0):
                            #     print 'wid, hei, dep = %.2f, %.2f, %.2f' % (
                            #         wid, hei, dep)
                            
                            if (not already_have):
                                # don't be empty (redundant now)
                                uniqueBoxes +=1
                                # print(hei,wid,dep)
                                if ((hei > 0) and 
                                    (wid > 0) and 
                                    (dep > 0) and
                                    # be less than huge
                                    (hei < 10.0) and 
                                    (wid < 10.0) and
                                    (dep < 10.0)):
                                    # be bigger than 2 vox 
                                    # (hei > 1.0) and 
                                        # print 'mean(y), min(y) max(y), ymin_, ymax_, ymin, ymax = %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (
                                        #     np.mean(y), np.min(y), np.max(y), ymin_, ymax_, ymin, ymax)
                                        # print 'xc, yc, zc = %.2f, %.2f, %.2f; wid, hei, dep = %.2f, %.2f, %.2f' % (
                                        #     xc, yc, zc, wid, hei, dep)
                                        
                                        # print 'wid, hei, dep = %.2f, %.2f, %.2f' % (wid, hei, dep)
                                        # # print 'theta = %.2f' % theta
                                        
                                        box = np.int0(box)
                                        cv2.drawContours(image,[box],-1,(0,191,255),1)

                                        boxes3D[boxcount,:] = box3D
                                        scores[boxcount] = np.random.uniform(0.1, 1.0)

                                        conn_ = np.zeros([H, W, D], np.float32)
                                        conn_[extracted_vox] = 1.0
                                        conn[boxcount] = conn_

                                        # imsave('boxes_%02d.png' % (boxcount), image)
                                        # print 'drawing box %d, with fake score %.2f' % (boxcount, scores[boxcount])
                                        # print box
                                        boxcount += 1
                                        box3D_list.append(box3D)
                            else:
                                pass
                        else:
                            print('box overflow; found more than %d' % N)
    return image, boxes3D, scores, conn


def compute_center_surround_score(summ_writer, boxes3D_camR, vox_coord_mem, sub_occ,ns):
    N=10
    
    B,K,_ = boxes3D_camR.shape
    scores = torch.zeros([B,K])
    protos_shape = vox_coord_mem.proto.shape
    MH2,MW2,MD2 = protos_shape
    MH,MW,MD = (MH2*2,MW2*2,MD2*2)

    halfmem_protos = torch.zeros([B]+protos_shape)

    obj_masks_1 = utils_vox.assemble_padded_obj_mask3D(boxes3D_camR, scores, halfmem_protos, vox_coord_mem, coeff=1.0)
    obj_masks_2 = utils_vox.assemble_padded_obj_mask3D(boxes3D_camR, scores, halfmem_protos, vox_coord_mem, coeff=1.25)
    # these are B x K x MH2 x MW2 x MD2 x 1
    # pad back up
    boxes3D_camR = F.pad(boxes3D_camR, [0, 0, 0, N - K, 0, 0])
    # the idea of a center-surround feature is:
    # there should be stuff in the center but not in the surround
    # so, i need the volume of the center
    # and the volume of the surround
    # then, the score is center-surround (minus)
    center_mask = obj_masks_1
    surround_mask = obj_masks_2 - obj_masks_1
    sub_mag = utils_basic.l2_on_axis(sub_occ, axis=4, keepdim=True)
    # flow_mag is B x MH2 x MW2 x MD2 x 1
    # sub_mag_mean, sub_mag_var = tf.nn.moments(sub_mag, [1, 2, 3])
    sub_mag_var, sub_mag_mean = torch.var_mean(sub_mag, [1, 2, 3])
    sub_mag_mean = sub_mag_mean.view(B, 1, 1, 1, 1)
    sub_mag_var = sub_mag_var.view(B, 1, 1, 1, 1)
    sub_mag_ = (sub_mag - sub_mag_mean) / (EPS + torch.sqrt(sub_mag_var))
    # sub_mag_ = tf.tile(tf.expand_dims(sub_mag_, axis=1), [1, K, 1, 1, 1, 1])
    sub_mag_ = sub_mag_.unsqueeze(1)
    sub_mag_ = sub_mag_.repeat(1, K, 1, 1, 1, 1)
    # sub_mag_ is B x 1 x MH2 x MW2 x MD2 x 1

    center_ = utils_basic.reduce_masked_mean(sub_mag_, center_mask, dim=[2, 3, 4, 5])
    surround_ = utils_basic.reduce_masked_mean(sub_mag_, surround_mask, dim=[2, 3, 4, 5])
    # st()
    scores = center_ - surround_
    # scores is B x K, with arbitrary range
    # scores = tf.nn.softmax(scores, axis=1)
    # scores is B x K, in range [0,1], and they conveniently sum to 1
    # scores = tf.pad(scores, [[0, 0], [0, N - K]])
    # Pytorch applies the padding on dimensions in reverse order. 
    scores = F.pad(scores, [0, N - K, 0, 0])
    # st()
    scores = utils_basic.l2_on_axis(scores,0)
    scores = utils_basic.l2_normalize(scores)
    for ind in range(K):
        center_mask_vis = torch.mean(center_mask[:, ind], dim=1)
        # center_mask_vis = tf.image.resize(center_mask_vis, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        center_mask_vis = torch.nn.functional.interpolate(center_mask_vis, size=[MW, MD], mode='nearest')
        summ_writer.summ_oned(ns + 'center_mask_%d' % ind, center_mask_vis, is3D=True)
    return scores

def get_boxes(summ_writer,ns,sub_occ,unp_,vox_coord_mem,origin_T_camXs,pix_T_cams,rgb_camXs,config):
    coord_mem = vox_coord_mem.coord
    vis_, boxes3D_mem, scores, conns = get_boxes_from_occ(sub_occ, unp_,coord_mem)
    B,N,_ =boxes3D_mem.shape
    B,MW2,MD2,_ =vis_.shape
    
    MW = MW2 * 2
    MD = MD2 * 2
    MH = config.MH
    MH2 = config.MH2

    #tids = tf.cast(torch.linspace(1.0, N, N), tf.int32)
    tids = torch.linspace(1.0, N, N).int()
    # tids = tf.tile(torch.reshape(tids, [1, -1]), [B, 1])
    tids = tids.view(1,-1).repeat(B, 1)

    # vis = tf.image.resize(vis_, [MW, MD],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vis = torch.nn.functional.interpolate(vis_, size=[MW, MD], mode='nearest')
    vis = utils_improc.preprocess_color(vis)
    # for now not needed
    # vis = (vis + unp_vis) / 2.0  # improve clarity a bit

    # visualize
    #TODO: Fix this import
    #utils.improc.summ_rgb(sc,ns + 'boxes_vis', vis, is3D=True)

    # boxes3D_mem is B x N x 9
    xc, yc, zc, lx, ly, lz, rx, ry, rz = torch.unbind(boxes3D_mem, dim=2)

    xz = torch.stack([xc, zc], dim=2)
    proto = torch.zeros([B, MW2, MD2, 1])
    centroid_mask = utils_improc.xy2mask(xz, proto.shape[0], proto.shape[1])

    # centroid_mask = tf.image.resize(centroid_mask, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    centroid_mask = torch.nn.functional.interpolate(centroid_mask, size=[MW, MD], mode='nearest')
    # visualize
    summ_writer.summ_oned(ns + 'centroid_mask', centroid_mask, is3D=True)

    corners_mem = utils_geom.transform_boxes_to_corners(boxes3D_mem)
    # corners_mem is B x N x 8 x 3
    # corners_mem = tf.reshape(corners_mem, [B, -1, 3])
    corners_mem = corners_mem.view(B, -1, 3)
    xc = corners_mem[:, :, 0]
    zc = corners_mem[:, :, 2]
    # in the raw image, z goes right and x goes down, so we do this:
    zx = torch.stack([zc, xc], axis=2)
    proto = torch.zeros([B, MW2, MD2, 1])
    corner_mask = utils_improc.xy2mask(zx, proto.shape[0], proto.shape[1])
    # corner_mask = tf.image.resize(corner_mask, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    corner_mask = torch.nn.functional.interpolate(corner_mask, size=[MW, MD], mode='nearest')
    # visualize

    summ_writer.summ_oned(ns + 'corner_mask', corner_mask, is3D=True)
    # unp_vis_a is B x MW x MD x 3
    # unp_ is B x MW2 x MD2 x 3
    #TODO: Uncomment below visualization after adding proper visualization function in utils_improc
    # o = utils.improc.draw_boxes3D_mem_on_mem(unp_, boxes3D_mem, scores, tids)
    # o = tf.image.resize(o, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # o = torch.nn.functional.interpolate(o, size=[MW, MD], mode='nearest')

    # visualize
    # summ_writer.summ_rgb(ns + 'boxes3D_on_mem', o, is3D=True)

    # i need to transform these mem boxes into refcam coords
    # as a wwarm up let's transform teh corners and vis those

    # ref_T_mem = utils.vox.get_ref_T_mem(B, MH2, MW2, MD2)


    corners_mem = utils_geom.transform_boxes_to_corners(boxes3D_mem)
    corners_mem_ = corners_mem.view(B, N * 8, 3)
    Z, Y, X = vox_coord_mem.proto.shape[0], vox_coord_mem.proto.shape[1], vox_coord_mem.proto.shape[2]
    corners_camR_ = utils_vox.Mem2Ref(corners_mem_, Z, Y, X)

    camX0_T_camRs = torch.inverse(origin_T_camXs[:,0])
    # camX1_T_camRs = torch.inverse(origin_T_camXs[:,1])

    corners_camX0_ = utils_geom.apply_4x4(camX0_T_camRs, corners_camR_)
    # corners_camX1_ = utils_geom.apply_4x4(camX1_T_camRs, corners_camR_)


    corners_camR = corners_camR_.view(B, N, 8, 3)
    corners_camX0 = corners_camX0_.view(B, N, 8, 3)
    # corners_camX1 = corners_camX1_.view(B, N, 8, 3)
    # corners_to_boxes ONLY works in R coords, since angle directions may change in other views
    boxes3D_camR = utils_geom.transform_corners_to_boxes(corners_camR)
    # visualize
    # st()
    summ_writer.summ_box_by_corners(ns + 'boxes3D_camX0', rgb_camXs[:, 0], corners_camX0, scores, tids,
                                     pix_T_cams[:, 0])
    # utils.improc.summ_box_by_corners(sc,ns + 'boxes3D_camX1', rgb_camXs[:, 1], corners_camX1, scores, tids,
    #                                  pix_T_cams[:, 1])

    conns = torch.unbind(conns, dim=1)
    # each conn is B x MH2 x MW2 x MD2 x 1

    # doing all N of them is too much
    K = 2
    halfmem_protos = torch.zeros([B, MH2, MW2, MD2])
    # st()
    boxes3D_camR = boxes3D_camR[:, :K]
    scores = scores[:, :K]

    obj_masks_1 = utils_vox.assemble_padded_obj_mask3D(boxes3D_camR, scores, halfmem_protos, vox_coord_mem, coeff=1.0)
    obj_masks_2 = utils_vox.assemble_padded_obj_mask3D(boxes3D_camR, scores, halfmem_protos, vox_coord_mem, coeff=1.5)
    # these are B x K x MH2 x MW2 x MD2 x 1
    # pad back up
    boxes3D_camR = torch.pad(boxes3D_camR, [0, 0, 0, N - K, 0, 0])

    # st()
    # self.rgb_gt
    # self.sub_e
    # the idea of a center-surround feature is:
    # there should be stuff in the center but not in the surround
    # so, i need the volume of the center
    # and the volume of the surround
    # then, the score is center-surround (minus)
    center_mask = obj_masks_1
    surround_mask = obj_masks_2 - obj_masks_1
    sub_mag = utils_basic.l2_on_axis(sub_occ, axis=4)
    # flow_mag is B x MH2 x MW2 x MD2 x 1

    # sub_mag_mean, sub_mag_var = tf.nn.moments(sub_mag, [1, 2, 3])
    sub_mag_var, sub_mag_mean = torch.var_mean(sub_mag, [1, 2, 3])
    sub_mag_mean = sub_mag_mean.view(B, 1, 1, 1, 1)
    sub_mag_var = sub_mag_var.view(B, 1, 1, 1, 1)
    sub_mag_ = (sub_mag - sub_mag_mean) / (EPS + torch.sqrt(sub_mag_var))
    # sub_mag_ = tf.tile(tf.expand_dims(sub_mag_, axis=1), [1, K, 1, 1, 1, 1])
    sub_mag_ = sub_mag_.unsqueeze(1)
    sub_mag_ = sub_mag_.repeat(1, K, 1, 1, 1, 1)
    # sub_mag_ is B x 1 x MH2 x MW2 x MD2 x 1

    center_ = utils_basic.reduce_masked_mean(sub_mag_, center_mask, dim=[2, 3, 4, 5])
    surround_ = utils_basic.reduce_masked_mean(sub_mag_, surround_mask, dim=[2, 3, 4, 5])

    scores = center_ - surround_
    # scores is B x K, with arbitrary range
    scores = F.softmax(scores.float(), dim=1)
    # scores is B x K, in range [0,1], and they conveniently sum to 1
    # scores = tf.pad(scores, [[0, 0], [0, N - K]])
    scores = torch.pad(scores, [0, N - K, 0, 0])

    cs = torch.mean(center_mask * sub_mag_ - surround_mask * sub_mag_, dim=5, keepdims=True)
    # cs is B x K x MH2 x MW2 x MD2 x 1
    for ind in range(K):
        conn = torch.mean(conns[ind], dim=1)
        # conn is B x MW2 x MD2 x 1
        # conn = tf.image.resize(conn, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conn = torch.nn.functional.interpolate(conn, size=[MW, MD], mode='nearest')

        # visualize
        summ_writer.summ_oned(ns + 'conn_%d' % ind, conn, is3D=True)

        center_mask_vis = torch.mean(center_mask[:, ind], dim=1)
        # center_mask_vis = tf.image.resize(center_mask_vis, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        center_mask_vis = torch.nn.functional.interpolate(center_mask_vis, size=[MW, MD], mode='nearest')

        # visualize
        summ_writer.summ_oned(ns + 'center_mask_%d' % ind, center_mask_vis, is3D=True)
        surround_mask_vis = torch.mean(surround_mask[:, ind], dim=1)
        # surround_mask_vis = tf.image.resize(surround_mask_vis, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        surround_mask_vis = torch.nn.functional.interpolate(surround_mask_vis, size=[MW, MD], mode='nearest')

        # visualize
        summ_writer.summ_oned(ns + 'surround_mask_%d' % ind, surround_mask_vis, is3D=True)
        cs_vis = torch.mean(cs[:, ind], dim=1)
        # print_shape(cs_vis)
        # cs_vis = tf.image.resize(cs_vis, [MW, MD], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cs_vis = torch.nn.functional.interpolate(cs_vis, size=[MW, MD], mode='nearest')

        # visualize
        summ_writer.summ_oned(ns + 'cs_%d' % ind, cs_vis, is3D=True)
        box3D = boxes3D_camR[:, ind:ind + 1]
        score = scores[:, ind:ind + 1]
        tid = tids[:, ind:ind + 1]
    return scores, boxes3D_camR

def visualize_hard_mining(posPair,topks,ranks,unpRs,visual2ds,current_index,mbr_unpr,negative=False):
    # mbr_unpr = cross_corr.meshgrid_based_rotation(32,32,32)    

    unpRs_e,unpRs_g = unpRs
    obs_visual_2d_e,obs_visual_2d_g = visual2ds

    topkImg,topkD,topkH,topkW,topkR  = topks

    query_index = posPair[current_index][0]
    e_index = posPair[current_index][1]
    g_index = posPair[current_index][2]

    chosen_patch_e = topkImg[int(query_index),int(e_index)]
    chosen_patch_g = topkImg[int(query_index),int(g_index)]

    pool_e_index = ranks[int(query_index),int(chosen_patch_e)]
    pool_g_index = ranks[int(query_index),int(chosen_patch_g)]

    visual2d_e = obs_visual_2d_e[pool_e_index]
    visual2d_g = obs_visual_2d_g[pool_g_index]
    
    visual2d_e = utils_improc.preprocess_color(torch.from_numpy(visual2d_e).unsqueeze(0).permute(0,3,1,2))
    visual2d_g = utils_improc.preprocess_color(torch.from_numpy(visual2d_g).unsqueeze(0).permute(0,3,1,2))

    unpR_e = unpRs_e[pool_e_index]
    unpR_g = unpRs_g[pool_g_index]

    _,unp_H,unp_W =list(unpR_e.shape)
    assert unp_H == unp_W

    unp_size = unp_H
    emb_size = hyp.BOX_SIZE

    scale_ratio =  unp_size//emb_size

    chosen_H_e = topkH[int(query_index),int(e_index)]
    chosen_W_e = topkW[int(query_index),int(e_index)]
    chosen_D_e = topkD[int(query_index),int(e_index)]
    chosen_R_e = topkR[int(query_index),int(e_index)]


    chosen_H_g = topkH[int(query_index),int(g_index)]
    chosen_W_g = topkW[int(query_index),int(g_index)]
    chosen_D_g = topkD[int(query_index),int(g_index)]
    chosen_R_g = topkR[int(query_index),int(g_index)]
    
    rotationInput = torch.stack([unpR_e, unpR_g], dim=0)
    rotatedOut = mbr_unpr.rotate2D(rotationInput)

    # Rotate the tensors
    unpR_e_rotated = rotatedOut[0, :, chosen_R_e.long()] 
    unpR_g_rotated = rotatedOut[1, :, chosen_R_g.long()]

    unpR_e_boxed = utils_improc.draw_rect_on_image(unpR_e_rotated,[chosen_W_e,chosen_D_e],scale_ratio,negative=negative)
    unpR_g_boxed = utils_improc.draw_rect_on_image(unpR_g_rotated,[chosen_W_g,chosen_D_g],scale_ratio,negative=negative)

    unpR_e = unpR_e.unsqueeze(0)
    unpR_g = unpR_g.unsqueeze(0)
    # st()

    ret_e = torch.cat([unpR_e_boxed, visual2d_e], dim=3)
    ret_g = torch.cat([unpR_g_boxed, visual2d_g], dim=3)

    return [ret_e,ret_g]

def visualize_eval_mining(top_g,selected_e,unp_ge,vis2d_ge,summ_writer,mbr_unpr):
    unpR_gs,unpR_e = unp_ge
    vis2D_gs,vis2D_e = vis2d_ge
    topkImg_i,topkD,topkH,topkW,topkR = top_g
    # D_g,H_g,W_g = top_g
    D_e, H_e, W_e  = selected_e
    # st
    unpR_e= (torch.from_numpy(unpR_e).permute(2,0,1))
    _,unp_H,unp_W =list(unpR_e.shape)
    assert unp_H == unp_W
    _,vis2D_H,vis2D_W =list(vis2D_e.shape)
    assert vis2D_H == unp_W

    unp_size = unp_H
    emb_size = hyp.BOX_SIZE

    scale_ratio =  unp_size//emb_size


    # st()
    # summ_writer.summ_rgb("debug/unp_before",unpR_e.unsqueeze(0))
    vis2D_e = utils_improc.preprocess_color(vis2D_e.unsqueeze(0))
    unpR_e_boxed = utils_improc.draw_rect_on_image(unpR_e,[W_e,D_e],scale_ratio,negative=False)
    unpR_e_boxed = torch.cat([unpR_e_boxed,vis2D_e],dim=2)
    # summ_writer.summ_rgb("debug/unp_before",unpR_e.unsqueeze(0))

    unp_gs_boxed = []
    for rank in range(10):
        pool_g_index_retrieved = topkImg_i[0,rank]
        W_top_g = topkW[0,rank]
        H_top_g = topkH[0,rank]
        D_top_g = topkD[0,rank]
        R_top_g = topkR[0,rank]
        unp_g_top = unpR_gs[pool_g_index_retrieved]
        vis2D_g_top = vis2D_gs[pool_g_index_retrieved]
        unpR_g = (torch.from_numpy(unp_g_top).permute(2,0,1)).unsqueeze(0)
        unpR_g_rotated = mbr_unpr.rotate2D(unpR_g.cuda()).cpu()
        unpR_g = unpR_g_rotated[:,:,R_top_g.to(torch.int32)].squeeze(0)
        unpR_g_boxed = utils_improc.draw_rect_on_image(unpR_g,[W_top_g,D_top_g],scale_ratio,negative=False)
        vis2D_g_top = utils_improc.preprocess_color(vis2D_g_top.unsqueeze(0))
        unpR_g_boxed = torch.cat([unpR_g_boxed,vis2D_g_top],dim=2)
        unp_gs_boxed.append(unpR_g_boxed)
    unps_boxed_e_gs = [unpR_e_boxed] + unp_gs_boxed
    return unps_boxed_e_gs




def sample_boxes(b_mask,num_objects,y_max=None,shape_val = None):
    y_act_max = float(y_max.cpu().numpy())
    y_max = round(y_act_max)
    bboxes = []
    if shape_val is None:
        print("add shape augmentation")
    for i in range(num_objects):
        shape_curr = list(shape_val[i])
        zsize,ysize,xsize = shape_curr
        intersect = True
        box = None
        while intersect:
            x_min = random.randint(0, (hyp.X2-xsize-1))
            z_min = random.randint(0, (hyp.Z2-zsize-1))
            x_max = x_min + xsize
            z_max = z_min + zsize
            # st()
            y_min = y_max - ysize
            # y_act_min = y_min
            # st()
            int_voxels = torch.sum(b_mask[z_min:z_max,y_min:y_max,x_min:x_max]).cpu().numpy()
            if int_voxels == 0:
                intersect = False
                box = [[x_min,y_min,z_min],[x_max,y_act_max,z_max]]
                b_mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1
        bboxes.append(box)
    return np.array(bboxes)

def check_fill_dict(contents,styles):
    if hyp.dataset_name == "real":
        is_filled_cont = len(list(contents.keys())) == 5
        is_filled_style = len(list(styles.keys())) == 15
    else:
        is_filled_cont = len(list(contents.keys())) == 3
        is_filled_style = len(list(styles.keys())) == 16
    is_num_ok_cont_1 = np.array([len(i)==hyp.few_shot_nums for i in contents.values()]).all()
    is_num_ok_cont_2 = np.array([len(i)==hyp.few_shot_nums for i in styles.values()]).all()
    return is_filled_cont and is_filled_style and is_num_ok_cont_1 and is_num_ok_cont_2

def crop_object_tensors(embs,boxes,scores):
    boxes = torch.clamp(boxes,min=0)    
    all_objects = []
    for index_batch,emb in enumerate(embs):
        for index_box,box in enumerate(boxes[index_batch]):
            if scores[index_batch][index_box] > 0:
                lower,upper = torch.unbind(box)
                xmin,ymin,zmin = [torch.round(i).to(torch.int32) for i in lower]
                xmax,ymax,zmax = [torch.round(i).to(torch.int32) for i in upper]
                # st()
                try:
                    object_tensor = embs[index_batch,:,zmin:zmax,ymin:ymax,xmin:xmax]
                except Exception as e:
                    print(e)
                all_objects.append(object_tensor)
    return all_objects

def create_object_tensors(embs,vis,boxes,scores,size,occs=False):
    num_embs = len(embs)
    embs = torch.stack(embs,axis=1)
    # over the batch
    boxes = torch.clamp(boxes,min=0)    
    all_objects = []
    if vis is not None:
        vis = torch.stack(vis,axis=1)
        all_objects_vis = []
    for index_batch,emb in enumerate(embs):
        for index_box,box in enumerate(boxes[index_batch]):
            if scores[index_batch][index_box] > 0:
                lower,upper = torch.unbind(box)
                xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
                xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]
                try:
                    object_tensor = embs[index_batch,:,:,zmin:zmax,ymin:ymax,xmin:xmax]
                    if occs:
                        object_tensor = torch.nn.functional.interpolate(object_tensor,size=size,mode='nearest')
                    else:
                        object_tensor = torch.nn.functional.interpolate(object_tensor,size=size,mode='trilinear')
                    if vis is not None:
                        object_visibility = vis[index_batch,:,:,zmin:zmax,ymin:ymax,xmin:xmax]
                        object_visibility = torch.nn.functional.interpolate(object_visibility,size=size,mode='nearest')
                except Exception as e:
                    object_tensor = torch.zeros_like(embs[index_batch,:,:,0:size[0],0:size[1],0:size[2]])
                    if vis is not None:
                        object_visibility = torch.zeros_like(vis[index_batch,:,:,0:size[0],0:size[1],0:size[2]])
                    print(e)
                all_objects.append(object_tensor)
                if vis is not None:
                    all_objects_vis.append(object_visibility)

    if len(all_objects) ==0:
        # st()
        embs = [[] for i in range(num_embs)]
        # st()
        if vis is not None:
            vis = [[] for i in vis]
        else:
            vis = [None]
        return_vals = embs + vis
    else:
        all_objects = torch.stack(all_objects)
        embs = torch.unbind(all_objects,dim=1)

        if vis is not None:
            all_objects_vis = torch.stack(all_objects_vis)
            vis = torch.unbind(all_objects_vis,dim=1)
            return_vals = embs + vis
        else:
            return_vals = embs  + tuple([None])
    return return_vals


def similarity_score(emb,emb1):
    B = 1
    vect_e = torch.nn.functional.normalize(torch.reshape(emb, [B, -1]),dim=1)
    vect_g = torch.nn.functional.normalize(torch.reshape(emb1, [B, -1]),dim=1)
    scores = torch.matmul(vect_e, vect_g.t())
    return scores.squeeze()

def create_object_tensors_filter_cs(embs,boxes,scores,size,cs_check=False):
    embs = torch.stack(embs,axis=1)    
    all_objects = []
    indices = []
    all_boxes = []
    all_neg_boxes = []    
    neg_indices = []
    for index_batch,emb in enumerate(embs):
        for index_box,box in enumerate(boxes[index_batch]):
            lower,upper = torch.unbind(box)
            xmin,ymin,zmin = [torch.floor(i).to(torch.int32) for i in lower]
            xmax,ymax,zmax = [torch.ceil(i).to(torch.int32) for i in upper]
            xdist,ydist,zdist = [(xmax-xmin),(ymax-ymin),(zmax-zmin)]
            if scores[index_batch][index_box] > 0 and xdist>0 and ydist>0 and zdist>0 and xmin>0 and ymin>0 and zmin>0 and  xmax<hyp.X2 and ymax<hyp.Y2 and zmax<hyp.Z2:
                try:
                    scores_similarity = []
                    object_tensor = embs[index_batch,:,:,zmin:zmax,ymin:ymax,xmin:xmax]
                    mean_score = 0
                    if cs_check:
                        # ztop
                        if zmax + zdist < hyp.Z2:
                            ztop_tensor = embs[index_batch,:,:,zmax:zmax + zdist,ymin:ymax,xmin:xmax]
                            score = similarity_score(object_tensor,ztop_tensor)
                            scores_similarity.append(score)
                            # we can crop this
                        # zbottom
                        if  zmin - zdist > 0:
                            zbottom_tensor = embs[index_batch,:,:,zmin - zdist:zmin,ymin:ymax,xmin:xmax]
                            score = similarity_score(object_tensor,zbottom_tensor)
                            scores_similarity.append(score)
                            # we can crop this
                        # ytop
                        if  ymax + ydist < hyp.Y2:
                            ytop_tensor = embs[index_batch,:,:,zmin:zmax,ymax:ymax + ydist,xmin:xmax]
                            score = similarity_score(object_tensor,ytop_tensor)
                            scores_similarity.append(score)
                            # we can crop this
                        # ybottom
                        if  ymin - ydist > 0:
                            ybottom_tensor = embs[index_batch,:,:,zmin:zmax, ymin - ydist:ymin,xmin:xmax]
                            score = similarity_score(object_tensor,ybottom_tensor)
                            scores_similarity.append(score)
                            # we can crop this
                        # xtop
                        if  xmax + xdist < hyp.X2:
                            xtop_tensor = embs[index_batch,:,:,zmin:zmax,ymin:ymax,xmax:xmax + xdist]
                            score = similarity_score(object_tensor,xtop_tensor)
                            scores_similarity.append(score)
                            # we can crop this
                        # xbottom
                        if  xmin - xdist > 0:
                            xbottom_tensor = embs[index_batch,:,:,zmin:zmax,ymin:ymax,xmin - xdist:xmin]
                            score = similarity_score(object_tensor,xbottom_tensor)
                            scores_similarity.append(score)                            
                            # we can crop this
                        if len(scores_similarity) == 0:
                            mean_score = 1.5
                            # if none of thesee abnove constrainsts meet then the box is too big! 
                            # so ignore it?
                        else:
                            mean_score = torch.stack(scores_similarity).mean()
                        # st()
                    if mean_score > hyp.neg_cs_thresh:
                        all_neg_boxes.append(box)
                        neg_indices.append([index_batch,index_box])
                    if mean_score < hyp.pos_cs_thresh:
                        object_tensor = torch.nn.functional.interpolate(object_tensor,size=size,mode='trilinear')
                        indices.append([index_batch,index_box])
                        all_objects.append(object_tensor)
                        all_boxes.append(box)
                except Exception as e:
                    object_tensor = torch.zeros_like(embs[index_batch,:,:,0:size[0],0:size[1],0:size[2]])
                    print(e)

    if len(neg_indices) > 0:
        neg_indices = torch.from_numpy(np.array(neg_indices))
        all_neg_boxes = torch.stack(all_neg_boxes)

    if len(all_objects) > 0:
        all_objects = torch.stack(all_objects)
        all_boxes = torch.stack(all_boxes)
        indices = torch.from_numpy(np.array(indices))
        embs = torch.unbind(all_objects,dim=1)
        if len(neg_indices) > 0:
            return_vals = embs   + tuple([indices]) + tuple([all_boxes]) +  tuple([neg_indices]) + tuple([all_neg_boxes])
        else:
            return_vals = embs   + tuple([indices]) + tuple([all_boxes]) +  tuple([None]) + tuple([None])
        return return_vals
    else:
        return [None,None,None,None,None,None]

def update_scene_with_objects(emb3D_scenes,emb3D_objects,boxes, scores):
    # emb3D_e_R, emb3D_g_R = emb3D_scene
    emb3D_scenes= emb3D_scenes.clone()
    emb3D_scenes = emb3D_scenes.unsqueeze(axis=1)
    sizes_val = [hyp.Z2,hyp.Y2,hyp.X2]
    if hyp.detach_background:
        emb3D_scenes = emb3D_scenes.detach()
    emb3D_objects = emb3D_objects.unsqueeze(axis=1)
    boxes = torch.clamp(boxes,min=0)
    index_to_take = -1
    for index_batch,emb_scene in enumerate(emb3D_scenes):
        for index_box,box in enumerate(boxes[index_batch]):
            if scores[index_batch][index_box] > 0:
                index_to_take += 1
                lower,upper = torch.unbind(box)
                lower = [torch.floor(i).to(torch.int32) for i in lower]
                upper = [torch.ceil(i).to(torch.int32) for i in upper]
                xmin,ymin,zmin = [max(i,0) for i in lower]

                xmax,ymax,zmax = [min(i,sizes_val[index]) for index,i in enumerate(upper)]
                size = [zmax-zmin,ymax-ymin,xmax-xmin]
                # print(index_to_take)
                try:
                    object_tensor = torch.nn.functional.interpolate(emb3D_objects[index_to_take],size=size,mode='nearest')
                    emb3D_scenes[index_batch,:,:,zmin:zmax,ymin:ymax,xmin:xmax] = object_tensor
                except Exception as e:
                    print("hello")
    emb3D_R = emb3D_scenes.squeeze(1)
    return emb3D_R


def update_scene_with_object_crops(emb3D_scenes,emb3D_objects,boxes, scores):
    # emb3D_e_R, emb3D_g_R = emb3D_scene
    emb3D_scenes= emb3D_scenes.clone()
    # emb3D_scenes = emb3D_scenes.unsqueeze(axis=1)
    sizes_val = [hyp.Z2,hyp.Y2,hyp.X2]
    # if hyp.detach_background:
    #     emb3D_scenes = emb3D_scenes.detach()
    # emb3D_objects = emb3D_objects.unsqueeze(axis=1)
    boxes = torch.clamp(boxes,min=0)
    index_to_take = -1
    for index_batch,emb_scene in enumerate(emb3D_scenes):
        for index_box,box in enumerate(boxes[index_batch]):
            if scores[index_batch][index_box] > 0:
                index_to_take += 1
                lower,upper = torch.unbind(box)
                lower = [torch.round(i).to(torch.int32) for i in lower]
                upper = [torch.round(i).to(torch.int32) for i in upper]
                xmin,ymin,zmin = [max(i,0) for i in lower]
                xmax,ymax,zmax = [min(i,sizes_val[index]) for index,i in enumerate(upper)]
                size = [zmax-zmin,ymax-ymin,xmax-xmin]
                # print(index_to_take)

                emb3D_scenes[index_batch,:,zmin:zmax,ymin:ymax,xmin:xmax] = emb3D_objects[index_to_take]
    # emb3D_R = emb3D_scenes.squeeze(1)
    return emb3D_scenes


def create_object_rgbs(rgbs,boxes_pix,scores):
    # over the batch
    boxes_pix = get_ends_of_corner(boxes_pix)
    boxes_pix = torch.clamp(boxes_pix,min=0)
    all_rgbs = []
    for index_batch,rgb in enumerate(rgbs):
        for index_box,box in enumerate(boxes_pix[index_batch]):
            if scores[index_batch][index_box] > 0:
                lower,upper = torch.unbind(box)
                xmin,ymin = torch.floor(lower).to(torch.int16)
                xmax,ymax = torch.ceil(upper).to(torch.int16)                
                # xmin,xmax,ymin,ymax = process_bounds([xmin,xmax,ymin,ymax])
                try:
                    object_rgb = rgb[:,ymin:ymax,xmin:xmax]
                    object_rgb = torch.squeeze(torch.nn.functional.interpolate(torch.unsqueeze(object_rgb,dim=0),size=[hyp.BOX_SIZE*2,hyp.BOX_SIZE*2],mode='bilinear'),dim=0)
                    # st()
                except Exception as e:
                    object_rgb = torch.zeros_like(rgb[:,0:hyp.BOX_SIZE*2,0:hyp.BOX_SIZE*2])
                    print(e)
                all_rgbs.append(object_rgb)
    try:
        if len(all_rgbs) == 0:
            all_rgbs = []
        else:
            all_rgbs = torch.stack(all_rgbs)
    except Exception as e:
        st()
        print("check")
    return all_rgbs

def create_object_classes(classes,filenames,scores):
    all_classes = []
    filename_g,filename_e = filenames
    all_filenames_g = []
    all_filenames_e = []
    for index_batch,class_val in enumerate(classes):
        for index_class,mini_class_val in enumerate(classes[index_batch]):
            if scores[index_batch][index_class] > 0:
                all_filenames_g.append(filename_g[index_batch])
                all_filenames_e.append(filename_e[index_batch])
                all_classes.append(mini_class_val)
    if len(all_classes) == 0:
        all_classes = []
        all_filenames_g = []
        all_filenames_e = []
    else:
        all_classes = np.stack(all_classes)
        all_filenames_g = np.stack(all_filenames_g)
        all_filenames_e = np.stack(all_filenames_e)
    return all_classes,[all_filenames_g,all_filenames_e]

def zero_out(featX,boxes,scores):
    featX_masks = []
    for index_batch,feat in enumerate(featX):
        feat_mask = torch.zeros_like(feat)
        for index_box,box in enumerate(boxes[index_batch]):
            if scores[index_batch][index_box] > 0:
                lower,upper = torch.unbind(box)
                xmin,ymin,zmin = [torch.floor(i).to(torch.int8) for i in lower]
                xmax,ymax,zmax = [torch.ceil(i).to(torch.int8) for i in upper]
                feat_mask[:,zmin:zmax,ymin:ymax,xmin:xmax] = torch.ones_like(feat_mask[:,zmin:zmax,ymin:ymax,xmin:xmax])
        featX_masks.append(feat_mask)
    featX_masks = torch.stack(featX_masks)
    featX  = featX * featX_masks
    return featX


def process_bounds(bounds):
    xmin,xmax,ymin,ymax = bounds
    xmin = torch.clamp(xmin,min=0)
    ymin = torch.clamp(ymin,min=0)
    return [xmin,xmax,ymin,ymax]

def get_ends_of_corner(boxes):
    min_box = torch.min(boxes,dim=2,keepdim=True).values
    max_box = torch.max(boxes,dim=2,keepdim=True).values
    boxes_ends = torch.cat([min_box,max_box],dim=2)
    return boxes_ends


def get_alignedboxes2thetaformat(aligned_boxes):
    B,N,_,_ = list(aligned_boxes.shape)
    aligned_boxes = torch.reshape(aligned_boxes,[B,N,6])
    B,N,_ = list(aligned_boxes.shape)
    xmin,ymin,zmin,xmax,ymax,zmax = torch.unbind(torch.tensor(aligned_boxes), dim=-1)
    xc = (xmin+xmax)/2.0
    yc = (ymin+ymax)/2.0
    zc = (zmin+zmax)/2.0
    w = xmax-xmin
    h = ymax - ymin
    d = zmax - zmin
    zeros = torch.zeros([B,N]).cuda()
    boxes = torch.stack([xc,yc,zc,w,h,d,zeros,zeros,zeros],dim=-1)
    return boxes

def bbox_rearrange(tree,boxes= [],classes={},all_classes=[]):
    for i in range(0, tree.num_children):
        updated_tree,boxes,classes,all_classes = bbox_rearrange(tree.children[i],boxes=boxes,classes=classes,all_classes=all_classes)
        tree.children[i] = updated_tree     
    if tree.function == "describe":
        if hyp.dataset_name=="bigbird":
            xmin,ymin,zmin,xmax,ymax,zmax = tree.bbox_camR_corners[0].astype(np.float32)
        else:
            xmax,ymax,zmin,xmin,ymin,zmax = tree.bbox_origin
        box = np.array([xmin,ymin,zmin,xmax,ymax,zmax])
        tree.bbox_origin = box
        boxes.append(box)
        classes["shape"] = tree.word
        all_classes.append(classes)
        classes = {}
    if tree.function == "combine":
        if "large" in tree.word or "small" in tree.word:
            classes["size"] = tree.word
        elif "metal" in tree.word or "rubber" in tree.word:
            classes["material"] = tree.word
        else:
            classes["color"] = tree.word
    return tree,boxes,classes,all_classes

def trees_rearrange(trees):
    updated_trees =[]
    all_bboxes = []
    all_scores = []
    all_classes_list = []
    for tree in trees:
        tree,boxes,_,all_classes = bbox_rearrange(tree,boxes=[],classes={},all_classes=[])
        if hyp.do_shape:
            classes = [class_val["shape"] for class_val  in all_classes]
        elif hyp.do_color:
            classes = [class_val["color"] for class_val  in all_classes]
        elif hyp.do_material:
            classes = [class_val["material"] for class_val  in all_classes]
        elif hyp.do_style:
            classes = [class_val["color"]+"_"+ class_val["material"] for class_val  in all_classes]
        elif hyp.do_style_content:
            classes = [class_val["shape"]+"/"+class_val["color"]+"_"+ class_val["material"] for class_val  in all_classes]
        elif hyp.do_color_content:            
            classes = [class_val["shape"]+"/"+class_val["color"] for class_val  in all_classes]
        elif hyp.do_material_content:            
            classes = [class_val["shape"]+"/"+ class_val["material"] for class_val  in all_classes]
        else:            
            classes = [class_val["shape"]+"/"+ class_val["color"] +"_"+class_val["material"] for class_val  in all_classes]
        boxes = np.stack(boxes)
        classes = np.stack(classes)
        N,_  = boxes.shape 
        assert N == len(classes)
        scores = np.pad(np.ones([N]),[0,hyp.N-N])
        boxes = np.pad(boxes,[[0,hyp.N-N],[0,0]])
        classes = np.pad(classes,[0,hyp.N-N])
        updated_trees.append(tree)
        all_classes_list.append(classes)
        all_scores.append(scores)
        all_bboxes.append(boxes)
    all_bboxes = np.stack(all_bboxes)
    all_scores = np.stack(all_scores)
    all_classes_list = np.stack(all_classes_list)
    return all_bboxes,all_scores,all_classes_list



def trees_rearrange_2d(trees):
    updated_trees =[]
    all_bboxes = []
    all_scores = []
    all_classes = []
    for tree in trees:
        tree,boxes,classes = bbox_rearrange(tree,boxes=[],classes=[])
        boxes = np.stack(boxes)
        
        boxes_from2d = tree.bbox2d_3d[:3]
        # st()
        boxes_from2d = get_ends_of_corner(boxes_from2d).squeeze(1)
        num_2d_boxes = boxes_from2d.shape[0]
        boxes_from2d = boxes_from2d.reshape([num_2d_boxes,6]).cpu().numpy()
        # st()
        boxes = boxes_from2d
        classes = np.stack(classes[:num_2d_boxes])
        N,_  = boxes.shape 
        scores = np.pad(np.ones([N]),[0,hyp.N-N])
        boxes = np.pad(boxes,[[0,hyp.N-N],[0,0]])
        classes = np.pad(classes,[0,hyp.N-N])
        updated_trees.append(tree)
        all_classes.append(classes)
        all_scores.append(scores)
        all_bboxes.append(boxes)
    all_bboxes = np.stack(all_bboxes)
    all_scores = np.stack(all_scores)
    all_classes = np.stack(all_classes)
    return all_bboxes,all_scores,all_classes

def bbox_rearrange_corners(tree,boxes= [],classes=[]):
    for i in range(0, tree.num_children):
        updated_tree,boxes,classes = bbox_rearrange_corners(tree.children[i],boxes=boxes,classes=classes)
        tree.children[i] = updated_tree     
    if tree.function == "describe":
        boxes.append(tree.bbox_origin)
        classes.append(tree.word)
    return tree,boxes,classes


def trees_rearrange_corners(trees):
    updated_trees =[]
    all_bboxes = []
    all_scores = []
    all_classes = []
    for tree in trees:
        tree,boxes,classes = bbox_rearrange_corners(tree,boxes=[],classes=[])
        boxes = np.stack(boxes)
        classes = np.stack(classes)
        N,num  = boxes.shape 
        assert num == 6
        scores = np.pad(np.ones([N]),[0,hyp.N-N])
        boxes = np.pad(boxes,[[0,hyp.N-N],[0,0]])
        classes = np.pad(classes,[0,hyp.N-N])
        updated_trees.append(tree)
        all_classes.append(classes)
        all_scores.append(scores)
        all_bboxes.append(boxes)
    all_bboxes = np.stack(all_bboxes)
    all_scores = np.stack(all_scores)
    all_classes = np.stack(all_classes)
    return all_bboxes,all_scores,all_classes