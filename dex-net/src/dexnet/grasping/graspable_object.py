#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: MrRen-sdhm

import os
import numpy as np
import open3d as o3d


class GraspableObjectO3d:
    """ Encapsulates geometric structures for computing contact in grasping.
    Attributes
    ----------
    cloud :
    normals :
    """
    def __init__(self, cloud, normals, cloud_voxel):
        self.cloud_ = np.asarray(cloud.points)
        self.normals_ = np.asarray(normals.points)
        self.cloud_voxel_ = np.asarray(cloud_voxel.points)
        self.cloud_o3d_ = cloud
        self.kd_tree_ = o3d.geometry.KDTreeFlann(cloud)

    @property
    def cloud(self):
        """
        cloud above the plane.
        type: nparray
        """
        return self.cloud_

    @property
    def normals(self):
        """
        normals of cloud above the plane.
        type: nparray
        """
        return self.normals_

    @property
    def cloud_voxel(self):
        """
        cloud contain the plane and voxeled.
        type: nparray
        usage: used to speed up collision check
        """
        return self.cloud_voxel_

    @property
    def cloud_o3d(self):
        """
        cloud above the plane as origin pcl formate.
        type: pcl PointCloud
        """
        return self.cloud_o3d_

    @property
    def kd_tree(self):
        """
        kd tree of the pcl PointCloud.
        """
        return self.kd_tree_

    def belong_to(self, point, r_ball=0.001):
        """
        Judging whether a point belongs to the cloud
        :return
            bool: if the point belong to cloud
            indice: the indice of the neibor point in cloud
        """
        [num, kd_indices, sqr_distances] = self.kd_tree_.search_radius_vector_3d(point, r_ball)
        # print(num, kd_indices, sqr_distances)
        if num > 0:  # find a neibor point
            return True, kd_indices[0]

        return False, None


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    file_dir = home_dir + "/Projects/GPD_PointNet/normals_estimator"

    """ o3d """
    if os.path.exists(file_dir + "/cloud.pcd"):
        cloud = o3d.io.read_point_cloud(file_dir + "/cloud.pcd")
        cloud_voxel = o3d.io.read_point_cloud(file_dir + "/cloud_voxel.pcd")
        normals = o3d.io.read_point_cloud(file_dir + "/normals_as_xyz.pcd")  # normals were saved as PointXYZ formate
    else:
        print("can't find any cloud or normals file!")
        raise NameError("can't find any cloud or normals file!")

    o3d_obj = GraspableObjectO3d(cloud, normals, cloud_voxel)

