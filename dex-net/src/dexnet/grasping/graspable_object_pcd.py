#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: MrRen-sdhm

import pcl
import numpy as np


class GraspableObjectPcd:
    """ Encapsulates geometric structures for computing contact in grasping.
    Attributes
    ----------
    cloud :
    normals :
    """
    def __init__(self, cloud, normals, cloud_voxel):
        self.cloud_ = cloud.to_array()
        self.normals_ = normals.to_array()
        self.cloud_voxel_ = cloud_voxel.to_array()
        self.cloud_pcl_ = cloud
        self.kd_tree_ = cloud.make_kdtree_flann()

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
    def cloud_pcl(self):
        """
        cloud above the plane as origin pcl formate.
        type: pcl PointCloud
        """
        return self.cloud_pcl_

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
        point = point.astype(np.float32)
        point_world_pcl = pcl.PointCloud(point.reshape(1, 3))
        kd_indices, sqr_distances = self.kd_tree_.radius_search_for_cloud(point_world_pcl, r_ball, 1)
        if sqr_distances[0, 0] != 0:  # find a neibor point
            return True, kd_indices[0, 0]

        return False, None

