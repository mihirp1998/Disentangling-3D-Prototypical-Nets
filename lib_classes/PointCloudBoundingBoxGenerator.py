import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
# import Imath, OpenEXR
import matplotlib.pyplot as plt 
import ipdb
# import imageio
import random
import socket
import os
# import gin
import copy
import pickle
#import open3d as o3d
from IPython import embed
from PIL import Image
import sys
import matplotlib as mpl
from os.path import join, isdir
import scipy
import math
from os import listdir
from matplotlib import pyplot
import glob
from itertools import permutations
from sklearn.mixture import BayesianGaussianMixture
from enum import Enum  
# import cc3d
from sklearn.cluster import DBSCAN as skdbscan
from collections import deque
import hyperparams as hyp

st = ipdb.set_trace
print(sys.path)


class bbox_detection_algorithms(Enum):
    VARIATIONAL_GAUSSIAN_MIXTURE = 1
    CONNECTED_COMPONENTS = 2
    CLUSTER_DBSCAN = 3

class BoundBoxGenerator:
    def __init__(self):
        print("Initialized bounding box generator")

    def make_pcd(self, pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        # if the dim is greater than 3 I expect the color
        if pts.shape[1] == 6:
            pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.\
                if pts[:, 3:].max() > 1. else pts[:, 3:])
        return pcd

    def merge_pcds(self, pcds):
        pts = [np.asarray(pcd.points) for pcd in pcds]
        # colors = [np.asarray(pcd.colors) for pcd in pcds]
        # assert len(pts) == 5, "these is the number of supplied pcd, it should match"
        combined_pts = np.concatenate(pts, axis=0)
        # combined_colors = np.concatenate(colors, axis=0)
        # assert combined_pts.shape[1] == 3, "concatenation is wrong"
        return combined_pts
 
    def visualize(self, list_of_pcds):
        # st()
        o3d.visualization.draw_geometries(list_of_pcds)
    def get_aggregated_pcd(self, pcds, extrinsics, vis=False, getTree = False):

        # you have the extrinsics and the pcds just rotate all the points and view
        recon_pcds = list()
        for pcd, ext in zip(pcds, extrinsics):
            
            # multiply with the corresponding extrinsics to transform to world space
            temp_pts = np.c_[pcd, np.ones(pcd.shape[0])]
            new_pts = np.dot(ext, temp_pts.T).T
            new_pcd = self.make_pcd(new_pts)
            recon_pcds.append(new_pcd)

            # visualize the pcds and move on
        
        combined_pts = self.merge_pcds(recon_pcds)
        aggregated_pcd = self.make_pcd(combined_pts)
        
        
        if vis:
            # print('view all pcds in camera frame')
            # self.visualize(recon_pcds)
            print('view all pcds in ar_tag frame')
            self.visualize([aggregated_pcd])
        return aggregated_pcd

    def subtract_point_clouds(self, scene_pcd, table_pcd, vis=False):
        pts_a = np.asarray(scene_pcd.points)
        pts_b = np.asarray(table_pcd.points)

        presumably_object_pts = pts_a - pts_b
        # actually I will form pcd with points which are not zero
        norm_new_pts = np.linalg.norm(presumably_object_pts, axis=1)
        # st()
        chosen_pts = pts_a[norm_new_pts>0.01]

        # clipped_pts = register_cam1_T_camX.get_inlier_pts(new_pts,
        #     clip_radius=0.5)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(chosen_pts[:, :3])
        if vis:
            # visualize_on_matplotlib(new_pcd)
            print("Visualize subtracted point cloud")
            o3d.visualization.draw_geometries([new_pcd])
        return new_pcd

    def get_bounding_box_coordinates(self, merged_pts):
        """Merges all the pcds computes the bbox and returns it
        """
        xmax, xmin = np.max(merged_pts[:, 0], axis=0),\
            np.min(merged_pts[:, 0], axis=0)

        ymax, ymin = np.max(merged_pts[:, 1], axis=0),\
            np.min(merged_pts[:, 1], axis=0)

        zmax, zmin = np.max(merged_pts[:, 2], axis=0),\
            np.min(merged_pts[:, 2], axis=0)

        return np.asarray([xmin, xmax, ymin, ymax, zmin, zmax])


    def form_eight_points_of_bbox(self, bbox_coords):
        xmin, ymin, zmin = bbox_coords[0:6:2]
        xmax, ymax, zmax = bbox_coords[1:6:2]
        eight_points = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin],\
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]]
        return eight_points


    def get_single_bbox_coords(self, merged_pts, vis=False):
        # merged_pts, merged_colors = merge_pcds(subt_pcd)
        # merged_pts, merged_colors = np.asarray(subt_pcd.points), np.asarray(subt_pcd.colors)
        bbox_coords= self.get_bounding_box_coordinates(merged_pts)
        if vis:
            # form the merged_pcd for visualization
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(merged_pts)
            # combined_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

            # points for linesets
            points = self.form_eight_points_of_bbox(bbox_coords)

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



    def cluster_using_dbscan(self, pcd, vis=False):
        pcd_points = np.asarray(pcd.points)
        clustering = skdbscan(eps=0.5, min_samples=2).fit(pcd_points)
        # st()
        clustered_pcd_indices = {}
        print("Number of objects found:", 1+np.max(clustering.labels_))
        for index, predicted_cluster in enumerate(clustering.labels_):
            if predicted_cluster not in clustered_pcd_indices:
                clustered_pcd_indices[predicted_cluster] = {'points': list()}
            clustered_pcd_indices[predicted_cluster]['points'].append(pcd_points[index])        
        
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=10.0, origin=[0, 0, 0])
        final_list = [pcd, mesh_frame]
        # final_list = [pcd]
        bbox_list = []
        for predicted_cluster in clustered_pcd_indices:
            ret, bbox_coords = self.get_single_bbox_coords(np.asarray(clustered_pcd_indices[predicted_cluster]['points']), vis)
            xmin,xmax,ymin,ymax,zmin,zmax = bbox_coords
            final_list.append(ret)
            bbox_list.append(np.asarray([xmin,ymin,zmin,xmax,ymax,zmax]))
        # if vis:
        #     print("Visualize bounding boxes from point cloud")
        #     o3d.visualization.draw_geometries(final_list)
        #The size of below np array will be Num_of_objects x 3 x 2
        bbox_array = np.stack(bbox_list)
        padded_bbox_array = np.zeros((hyp.MAX_OBJECTS_IN_SCENE, 6))
        #Multiply x and y coords by -1 to bring them to discovery/GoennTf2's coordinate system
        padded_bbox_array[:, :4] *= -1
        padded_bbox_array[: bbox_array.shape[0]] = bbox_array
        return padded_bbox_array

    def find_bounding_boxes(self, pcd, algo,vis=False):
        if algo == bbox_detection_algorithms.CLUSTER_DBSCAN:
            return self.cluster_using_dbscan(pcd,vis=vis)

    def get_bounding_boxes(self, inputs,vis=False):
        empty_pcd_np = inputs['empty_xyz_camXs'].numpy()
        extrinsics_np = inputs['origin_T_camXs'].numpy()
        scene_pcd_np = inputs['xyz_camXs'].numpy()
        
        #All empty scene pcds are expected to be same. Just take one entry from batch.
        print("Calculating point cloud for empty scene")
        aggregated_empty_pcd = self.get_aggregated_pcd(empty_pcd_np[0], extrinsics_np[0])

        aggregated_scene_pcd_list = []
        bounding_boxes_list = [] 
        #Loop over all items in the batch.
        for i in range(scene_pcd_np.shape[0]):
            print("Creating boudning boxes for scene: ", str(i))
            aggregated_scene_pcd = self.get_aggregated_pcd(scene_pcd_np[i], extrinsics_np[i])
            print("Aggregated pcd shape: ", aggregated_scene_pcd)
            # st()
            aggregated_scene_pcd_list.append(aggregated_scene_pcd)
            subtracted_pcd = self.subtract_point_clouds(aggregated_scene_pcd, aggregated_empty_pcd)


            bounding_boxes = self.find_bounding_boxes(subtracted_pcd, bbox_detection_algorithms.CLUSTER_DBSCAN,vis=vis)
            bounding_boxes_list.append(bounding_boxes)
            # st()
        # The shape of this array will be BatchSize x 10 x 6. 
        # We are assuming that there will be max 10 objects in a given scene.
        all_bboxes = np.stack(bounding_boxes_list)
        return all_bboxes,subtracted_pcd


