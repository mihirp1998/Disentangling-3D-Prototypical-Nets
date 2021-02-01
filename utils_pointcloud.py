import numpy as np 

import cv2
#import open3d as o3d
from PIL import Image
import torch
import utils_geom
from lib_classes import Nel_Utils as nlu
import matplotlib.pyplot as plt
import ipdb 
st = ipdb.set_trace
import copy
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.2, origin=[0, 0, 0])
def parse_intrinsics(intrinsics_mat):
    fx = intrinsics_mat[0][0]
    fy = intrinsics_mat[1][1]
    cx = intrinsics_mat[0][2]
    cy = intrinsics_mat[1][2]
    return fx, fy, cx, cy

def visualize_colored_pcd(depth, rgb, pix_T_cams):
    depth = depth.astype(np.float32)
    fx, fy, cx, cy = parse_intrinsics(pix_T_cams)

    # form the mesh grid
    xv, yv = np.meshgrid(np.arange(depth.shape[1], dtype=float), np.arange(depth.shape[0], dtype=float))

    xv -= cx
    xv /= fx
    xv *= depth
    yv -= cy
    yv /= fy
    yv *= depth
    points = np.c_[xv.flatten(), yv.flatten(), depth.flatten()]

    if rgb is not None:
        # flatten it and add to the points
        rgb = rgb.reshape(-1, 3)

    points = np.concatenate((points, rgb), axis=1)
    pcd = nlu.make_pcd(points)
    # if visualize:
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=.3, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, mesh_frame])
    
def draw_colored_pcd(pointcloud, rgbArray):
    # pointcloud = pointcloud.squeeze(0).numpy()
    pcd = nlu.make_pcd(pointcloud)
    if rgbArray.ndim == 2:
        pcd.colors = o3d.utility.Vector3dVector(rgbArray / 255.)
    else:
        pcd.colors = o3d.utility.Vector3dVector(rgbArray.reshape(-1,3) / 255.)
    return pcd

def create_pointcloud(depArray, rgbArray, pix_T_cams, visualize_color=False):
    
    pix_T_camX = torch.tensor(pix_T_cams).unsqueeze(0)
    depth_camXs = torch.tensor(depArray).unsqueeze(0).unsqueeze(0)
    if visualize_color:
        visualize_colored_pcd(depArray, rgbArray, pix_T_camX.squeeze(0).numpy())
    xyz_camXs = utils_geom.depth2pointcloud_cpu(depth_camXs.float(), pix_T_camX.float())
    pointcloud = xyz_camXs.squeeze(0).numpy()
    pcd = nlu.make_pcd(pointcloud)
    # o3d.visualization.draw_geometries([pcd, mesh_frame])
    return xyz_camXs, pcd

'''
rgb_camX - (240, 320, 3)
pix_T_cams - (4, 4)
bboxes - (1, 6)

# defining the corners as: clockwise backcar face, clockwise frontcar face:
#   E -------- F
#  /|         /|
# A -------- B .
# | |        | |
# . H -------- G
# |/         |/
# D -------- C

'''
def draw_boxes_on_rgb(rgb_camX, pix_T_cams, bboxes, visualize=False):
    xmin, ymin, zmin = bboxes[0, 0:3]
    xmax, ymax, zmax = bboxes[0, 3:6]
    rgb = np.copy(rgb_camX)
    bbox_xyz = np.array([[xmin, ymin, zmin], 
                        [xmin, ymin, zmax], 
                        [xmin, ymax, zmin],
                        [xmin, ymax, zmax],
                        [xmax, ymin, zmin], 
                        [xmax, ymin, zmax], 
                        [xmax, ymax, zmin],
                        [xmax, ymax, zmax]]
    )

    # bbox_xyz_pytorch = torch.from_numpy(bbox_xyz).unsqueeze(0).unsqueeze(0)
    # bbox_xyz_pytorch = torch.tensor(bbox_xyz_pytorch, dtype=torch.float32)
    # scores_pytorch = torch.ones((1,1), dtype=torch.uint8)
    # # st()
    # tids_pytorch = torch.ones_like(scores_pytorch)
    # rgb_pytorch = torch.tensor(torch.from_numpy(rgb_camX).permute(2, 0, 1).unsqueeze(0), dtype=torch.float32)
    # pix_T_cams_pytorch = torch.tensor(torch.from_numpy(pix_T_cams).unsqueeze(0), dtype=torch.float32)
    # summwriter = utils_improc.Summ_writer(None, 10, None, 8, 8)
    # # st()
    # bbox_rgb = summwriter.summ_box_by_corners("name_dummy", rgb_pytorch, bbox_xyz_pytorch, scores_pytorch, tids_pytorch, pix_T_cams_pytorch, only_return=True)
    # bbox_rgb = utils_improc.back2color(bbox_rgb).permute(0, 2, 3, 1)[0].numpy()
    # # st()
    # plt.imshow(bbox_rgb)
    # plt.show(block=True)
    
    bbox_img_xy = utils_geom.apply_pix_T_cam(torch.from_numpy(pix_T_cams).unsqueeze(0), 
                                    torch.from_numpy(bbox_xyz).unsqueeze(0)).squeeze(0) # torch.Size([8, 2])
    
    
    bbox_img_xy = bbox_img_xy.numpy()
    bbox_img_xy = bbox_img_xy.astype(int)
    A, E, D, H, B, F, C, G = bbox_img_xy

    A = (A[0], A[1])
    B = (B[0], B[1])
    C = (C[0], C[1])
    D = (D[0], D[1])
    E = (E[0], E[1])
    F = (F[0], F[1])
    G = (G[0], G[1])
    H = (H[0], H[1])
    

    lineThickness = 2
    # img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    rgb_camX = cv2.line(rgb_camX, A, E,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, E, H,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, D, H,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, D, A,(255,0,0),lineThickness)

    rgb_camX = cv2.line(rgb_camX, B, F,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, G, F,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, G, C,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, C, B,(255,0,0),lineThickness)

    rgb_camX = cv2.line(rgb_camX, A, B,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, E, F,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, C, D,(255,0,0),lineThickness)
    rgb_camX = cv2.line(rgb_camX, G, H,(255,0,0),lineThickness)
    # cv2.line(rgb_camX, bbox_img_xy[0], bbox_img_xy[2],(255,0,0),5)
    # st()
    # cv2.imshow('image',rgb_camX)
    if visualize:
        rgb = np.concatenate([rgb, rgb_camX], axis=1)
        plt.imshow(rgb)
        plt.show(block=True)
    return rgb_camX
    # st()
    # pass

def subtract_pointclouds(dict_a, dict_b, vis=True):
    pts_a = dict_a['points']
    pts_b = dict_b['points']

    presumably_object_pts = pts_a - pts_b
    # actually I will form pcd with points which are not zero
    norm_new_pts = np.linalg.norm(presumably_object_pts, axis=1)
    chosen_pts = pts_a[presumably_object_pts[:, 2] > 0.009]
    chosen_colors = dict_a['colors'][presumably_object_pts[:, 2] > 0.009]

    new_pts = np.c_[chosen_pts, chosen_colors]
    clipped_pts = get_inlier_pts(new_pts,
        clip_radius=0.5)

    assert clipped_pts.shape[1] == 6, "no color in the points, not acceptable"

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(clipped_pts[:, :3])
    new_pcd.colors = o3d.utility.Vector3dVector(clipped_pts[:, 3:]/255.0)
    if vis:
        o3d.visualization.draw_geometries([new_pcd])
    # st()
    return new_pcd

def vis_points(xyz,rgb):        
    new_pcd = o3d.geometry.PointCloud()
    xyz = xyz.reshape([-1,3])
    rgb = rgb.reshape([-1,3])
    # st()
    pts = get_inlier_pts(np.concatenate([xyz,rgb],axis=1),clip_radius=1.0)
    new_pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
    new_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:]/255.0)
    o3d.visualization.draw_geometries([new_pcd])
    return new_pcd



def get_inlier_pts(pts, clip_radius=5.0):
    """
    Assumptions points are centered at (0,0,0)
    only includes the points which falls inside the desired radius
    :param pts: a numpy.ndarray of form (N, 3)
    :param clip_radius: a float
    :return: numpy.ndarray of form (Ninliers, 3)
    """
    # do the mean centering of the pts first, deepcopy for this
    filter_pts = copy.deepcopy(pts[:, :3])
    mean_pts = filter_pts.mean(axis=0)
    assert mean_pts.shape == (3,), "wrong mean computation"
    filter_pts -= mean_pts

    sq_radius = clip_radius ** 2
    pts_norm_squared = np.linalg.norm(filter_pts, axis=1)
    idxs = (pts_norm_squared - sq_radius) <= 0.0
    chosen_pts = pts[idxs]
    return chosen_pts


def truncate_pcd_outside_bounds(bounds, xyz_camX):
    less_than_xmin = np.where(xyz_camX[:, 0]<bounds[0])
    more_than_xmax = np.where(xyz_camX[:, 0]>bounds[3])
    
    less_than_ymin = np.where(xyz_camX[:, 1]<bounds[1])
    more_than_ymax = np.where(xyz_camX[:, 1]>bounds[4])

    less_than_zmin = np.where(xyz_camX[:, 2]<bounds[2])
    more_than_zmax = np.where(xyz_camX[:, 2]>bounds[5])


    invalids = np.hstack((less_than_xmin, more_than_xmax, less_than_ymin, more_than_ymax, less_than_zmin, more_than_zmax))
    invalids = np.unique(invalids)

    mask = np.arange(xyz_camX.shape[0])
    mask = np.setdiff1d(mask, invalids)
    xyz_camX = xyz_camX[mask]
    return xyz_camX