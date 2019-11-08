#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: MrRen-sdhm

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import random
import time
import pcl

from autolab_core import RigidTransform
from dexnet.grasping.graspable_object_pcd import GraspableObjectPcd
from dexnet.grasping.grasp_pcd import ParallelJawPtGrasp3D

import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


try:
    import mayavi.mlab as mlab
except:
    try:
        import mayavi.mlab as mlab
    except ImportError:
        mlab = []
        logging.error('Failed to import mayavi')


# class GraspSampler(metaclass=ABCMeta):
class GraspSamplerPcd:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.

    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        self.friction_coef = config['sampling_friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        self.num_grasp_rots = config['num_grasp_rots']
        if 'max_num_surface_points' in list(config.keys()):
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 300
        if 'grasp_dist_thresh' in list(config.keys()):
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0

    @abstractmethod
    def sample_grasps(self, graspable, num_grasps_generate, vis, **kwargs):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.

        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        num_grasps_generate : int
        vis : bool
        """
        grasp = []
        return grasp
        # pass

    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : 目标返回抓取姿态数, 即满足条件的抓取姿态数
        grasp_gen_mult : sample_grasps的目标采样个数＝target_num_grasps*grasp_gen_mult, 即允许采样的个数, 之后还要筛选
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        vis : bool
            whether show the grasp on picture

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # get num grasps
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps

        print("[INFO] generate grasps... target_num_grasps:", target_num_grasps)

        if 'random_approach_angle' in kwargs:
            print("[INFO] use random_approach_angle:", kwargs['random_approach_angle'])

        grasps = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:  # 循环采样直到有 target_num_grasps 个抓取姿态, 或超过循环次数
            print("num_grasps_remaining:", num_grasps_remaining)
            # SAMPLING: generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining  # NOTE: Unused in AntipodalGraspSampler!
            new_grasps = self.sample_grasps(graspable, num_grasps_generate, vis, **kwargs)  # 通过力闭合进行抓取姿态采样

            # COVERAGE REJECTION: prune grasps by distance 通过距离删减抓取姿态
            pruned_grasps = []
            for grasp in new_grasps:
                min_dist = np.inf
                for cur_grasp in grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                for cur_grasp in pruned_grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= self.grasp_dist_thresh_:
                    pruned_grasps.append(grasp)

            for grasp in new_grasps:
                pruned_grasps.append(grasp)

            # ANGLE EXPANSION sample grasp rotations around the axis
            candidate_grasps = []
            if sample_approach_angles:
                for grasp in pruned_grasps:
                    # construct a set of rotated grasps
                    for i in range(self.num_grasp_rots):
                        rotated_grasp = copy.copy(grasp)
                        delta_theta = 0
                        print("This function can not use yet, as delta_theta is not set!")
                        rotated_grasp.set_approach_angle(i * delta_theta)
                        candidate_grasps.append(rotated_grasp)
            else:
                candidate_grasps = pruned_grasps

            # add to the current grasp set
            grasps += candidate_grasps
            logger.info('%d/%d grasps found after iteration %d.', len(grasps), target_num_grasps, k)

            grasp_gen_mult *= 2
            num_grasps_remaining = target_num_grasps - len(grasps)
            print("[INFO] Have generated grasps at present:", len(grasps))
            k += 1

        # shuffle computed grasps
        random.shuffle(grasps)
        if len(grasps) > target_num_grasps:
            logger.info('Truncating %d grasps to %d.', len(grasps), target_num_grasps)
            grasps = grasps[:target_num_grasps]
        logger.info('Found %d grasps.', len(grasps))
        return grasps

    @staticmethod
    def get_color(color):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'y':  # yellow
            color_f = (1, 1, 0)
        elif color == 'p':  # purple
            color_f = (1, 0, 1)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        return color_f

    def show_points(self, point, color='lb', scale_factor=.002):
        # if not isinstance(point, list):
        point = np.array(point)

        color_f = self.get_color(color)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        color_f = self.get_color(color)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    def show_arrow(self, point, direction, color='lb', scale_factor=.03):
        color_f = self.get_color(color)
        mlab.quiver3d(point[0], point[1], point[2], direction[0], direction[1], direction[2],
                      scale_factor=scale_factor, line_width=0.05, color=color_f, mode='arrow')

    def show_origin(self):
        self.show_arrow([0, 0, 0], [1, 0, 0], 'r', 0.1)
        self.show_arrow([0, 0, 0], [0, 1, 0], 'g', 0.1)
        self.show_arrow([0, 0, 0], [0, 0, 1], 'b', 0.1)

    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc, scale_factor=0.001):

        # un1 = grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width
        un2 = grasp_bottom_center
        # un3 = grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width
        # un4 = grasp_bottom_center
        # un5 = grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width
        # un6 = grasp_bottom_center
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        # self.show_points(un1, scale_factor=scale_factor * 4)
        # self.show_points(un3, scale_factor=scale_factor * 4)
        # self.show_points(un5, scale_factor=scale_factor * 4)
        # self.show_line(un1, un2, color='g', scale_factor=scale_factor)  # binormal/ major pc
        # self.show_line(un3, un4, color='b', scale_factor=scale_factor)  # minor pc
        # self.show_line(un5, un6, color='r', scale_factor=scale_factor)  # approach normal
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
                      scale_factor=.03, line_width=0.25, color=(0, 1, 0), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
                      scale_factor=.03, line_width=0.1, color=(0, 0, 1), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
                      scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')

    def get_hand_points(self, grasp_bottom_center, approach_normal, binormal):
        hh = self.gripper.hand_height
        fw = self.gripper.finger_width
        hod = self.gripper.hand_outer_diameter
        hd = self.gripper.hand_depth
        open_w = hod - fw * 2
        minor_pc = np.cross(approach_normal, binormal)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * hd + p5
        p2 = approach_normal * hd + p6
        p3 = approach_normal * hd + p7
        p4 = approach_normal * hd + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p

    def show_grasp_3d(self, hand_points, color='lb'):
        color_f = self.get_color(color)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        # mlab.points3d(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], color=(0, 0, 1), scale_factor=0.005)
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles, color=color_f, opacity=0.6)

    def check_collision_square(self, grasp_bottom_center, normal, major_pc, minor_pc, graspable, hand_p, way):
        normal_,  major_pc_, minor_pc_ = (normal, major_pc, minor_pc)
        normal = normal.reshape(1, 3)
        normal = normal / np.linalg.norm(normal)
        major_pc = major_pc.reshape(1, 3)
        major_pc = major_pc / np.linalg.norm(major_pc)
        minor_pc = minor_pc.reshape(1, 3)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        matrix = np.hstack([normal.T, major_pc.T, minor_pc.T])
        grasp_matrix = matrix.T  # same as cal the inverse
        if isinstance(graspable, GraspableObjectPcd):
            points = graspable.cloud_voxel
        else:
            raise ValueError("graspable must be GraspableObjectPcd")

        points = points - grasp_bottom_center.reshape(1, 3)
        tmp = np.dot(grasp_matrix, points.T)
        points_g = tmp.T
        if way == "p_open":
            s1, s2, s4, s8 = hand_p[1], hand_p[2], hand_p[4], hand_p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = hand_p[9], hand_p[1], hand_p[10], hand_p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = hand_p[2], hand_p[13], hand_p[3], hand_p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = hand_p[11], hand_p[15], hand_p[12], hand_p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

        # if has_p and way == "p_open":
        #     print("points_in_area", way, len(points_in_area))

        if False:
            print("points_in_area", way, len(points_in_area))
            mlab.figure(bgcolor=(1, 1, 1), size=(640, 480))
            mlab.clf()
            self.show_origin()
            self.show_grasp_norm_oneside(grasp_bottom_center, normal_, major_pc_, minor_pc_, scale_factor=0.001)
            self.show_points(points_g, scale_factor=.003)
            self.show_grasp_3d(hand_p)

            hand_p_origin = self.get_hand_points(grasp_bottom_center, normal, major_pc)
            self.show_grasp_3d(hand_p_origin, 'r')
            self.show_points(graspable.cloud_voxel, 'r', .003)
            self.show_points(graspable.cloud, 'b', .003)

            if len(points_in_area) != 0:
                self.show_points(points_g[points_in_area], color='r', scale_factor=.005)
            mlab.show()

        return has_p, points_in_area

    def show_all_grasps(self, grasps_for_show):

        for grasp_ in grasps_for_show:
            grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
            approach_normal = grasp_[1]
            binormal = grasp_[2]
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, binormal)
            self.show_grasp_3d(hand_points)

    def show_surface_points(self, graspable, color='lb', scale_factor=.002):
        surface_points = graspable.cloud
        self.show_points(surface_points, color=color, scale_factor=scale_factor)

    def check_collide(self, grasp_bottom_center, approach_normal, binormal, minor_pc, graspable, hand_points):
        bottom_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                    binormal, minor_pc, graspable, hand_points, "p_bottom")
        if bottom_points[0]:
            return True

        left_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                  binormal, minor_pc, graspable, hand_points, "p_left")
        if left_points[0]:
            return True

        right_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                   binormal, minor_pc, graspable, hand_points, "p_right")
        if right_points[0]:
            return True

        return False

    @staticmethod
    def get_surface_normal(graspable, selected_surface_pc, r_ball=0.005, rball_point_num=10):
        """ use kdtree to find neibor point's normal """

        # kd = graspable.cloud_pcl.make_kdtree_flann()
        selected_surface_pc = pcl.PointCloud(selected_surface_pc.reshape(1, 3))
        kd_indices, sqr_distances = graspable.kd_tree.radius_search_for_cloud(selected_surface_pc, r_ball, rball_point_num)
        normal = None
        for i in range(len(kd_indices[0])):
            if sqr_distances[0, i] != 0:
                normal = graspable.normals[kd_indices[0, i]]
                if np.isnan(np.sum(normal)):  # judge if has nan in normal
                    continue
                if np.linalg.norm(normal) != 0:
                    normal /= np.linalg.norm(normal)
                    break

        if normal is None:
            return None, None, None

        major_pc = np.array([normal[1], -normal[0], 0])
        if np.linalg.norm(major_pc) == 0:
            major_pc = np.array([1, 0, 0])
        major_pc = major_pc / np.linalg.norm(major_pc)
        minor_pc = np.cross(normal, major_pc)

        return normal, major_pc, minor_pc

    def display_grasps3d(self, grasps, color='lb'):
        """
        display ParrallelJawPtGraspPcd
        """
        color_f = self.get_color(color)
        for grasp in grasps:
            approach_normal = grasp.rotated_full_axis[:, 0]
            approach_normal = approach_normal / np.linalg.norm(approach_normal)
            major_pc = grasp.configuration[3:6]
            major_pc = major_pc / np.linalg.norm(major_pc)
            # minor_pc = np.cross(approach_normal, major_pc)
            center_point = grasp.center
            # print("grasps.center", grasp.center, grasp.center.shape)
            grasp_bottom_center = -self.gripper.hand_depth * approach_normal + center_point
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
            # print("hand_points:", hand_points)

            self.show_grasp_3d(hand_points, color=color_f)

    @staticmethod
    def new_window( size=500):
        mlab.figure(bgcolor=(0.5, 0.5, 0.5), size=(size, size))

    @staticmethod
    def show():
        mlab.show()


class GpgGraspSamplerPcd(GraspSamplerPcd):
    """
    Sample grasps by GPG.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, graspable, num_grasps, max_num_samples=30, vis=False, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObjectPcd`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_dy': 5,  # number 10
            'num_dz': 1,  # number 5
            'dtheta': 5,  # unit degree 5
            'range_dtheta_normal': 45,
            'range_dtheta_minor': 45,
            'range_dtheta_major': 45,
            'approach_step': 0.01,
            'keepaway_step': 0.015,
            'min_points_num': 50,  # min voxeled points num in open area

            'rball_point_num': 10,
            'r_ball': 0.005,
            'show_point_normals': False,
        }

        # start the time
        start = time.perf_counter()

        # get all surface points
        all_points = graspable.cloud

        # construct point cloud and voxel grid
        voxel = graspable.cloud_pcl.make_voxel_grid_filter()
        voxel.set_leaf_size(0.02, 0.02, 0.02)
        surface_points = voxel.filter().to_array()

        # filter points by axis x
        min_x = 0.002
        selected_indices = np.where(surface_points[:, 0] > min_x)[0]
        surface_points = surface_points[selected_indices]

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []
        grasp_test = []

        # visualize selected surface points and normals
        if False:
            for i in range(max_num_samples):
                # get candidate contacts
                ind = np.random.choice(num_surface, size=1, replace=False)
                selected_point = surface_points[ind, :].reshape(3)
                normal, major_pc, minor_pc = self.get_surface_normal(graspable, selected_point, 0.005, )
                if normal is None:
                    continue

                if normal[0] == np.nan:
                    print("[ERRO] normal == np.nan")
                print("[DEBUG]", normal, major_pc, minor_pc)
                self.show_points(selected_point, 'r', 0.005)
                self.show_arrow(selected_point, normal, 'r')
                self.show_arrow(selected_point, major_pc, 'g')
                self.show_arrow(selected_point, minor_pc, 'b')
            self.show_points(surface_points)
            self.show()

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            if params['show_point_normals']:
                self.show_points(selected_surface, 'r')

            """ cal local frame: normal, major_pc, minor_pc """
            r_ball = params['r_ball']
            rball_point_num = params['rball_point_num']
            normal, major_pc, minor_pc = self.get_surface_normal(graspable, selected_surface, r_ball, rball_point_num)
            if normal is None:
                continue

            # show local coordinate
            if params['show_point_normals']:
                self.show_arrow(selected_surface, normal, 'r')
                self.show_arrow(selected_surface, major_pc, 'g')
                self.show_arrow(selected_surface, minor_pc, 'b')

            """ Step1: rotat grasp around an axis(minor_pc:blue) """
            potential_grasp = []
            for dtheta in np.arange(-params['range_dtheta_minor'], params['range_dtheta_minor'] + 1, params['dtheta']):
                dy_potentials = []
                x, y, z = minor_pc
                rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))

                """ Step2: move step by step according to major_pc """
                for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                    (params['num_dy'] + 1) * self.gripper.finger_width, self.gripper.finger_width):
                    # compute centers and axes
                    tmp_major_pc = np.dot(rotation, major_pc)
                    tmp_grasp_normal = np.dot(rotation, normal)
                    tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                    # go back a bite after rotation dtheta and translation dy!
                    tmp_grasp_bottom_center = self.gripper.init_bite * (-tmp_grasp_normal) + tmp_grasp_bottom_center

                    has_open_points, points_in_open = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                                  tmp_major_pc, minor_pc, graspable,
                                                                                  hand_points, "p_open")
                    bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                   tmp_major_pc, minor_pc, graspable,
                                                                   hand_points, "p_bottom")

                    # grasp_test.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                    #                               tmp_major_pc, minor_pc, tmp_grasp_bottom_center])
                    #
                    # self.show_arrow(tmp_grasp_bottom_center, tmp_grasp_normal, 'r')
                    # self.show_all_grasps(grasp_test)
                    # self.show_points(all_points)
                    # mlab.show()

                    if len(points_in_open) > params['min_points_num'] and bottom_points is False:
                        left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_left")
                        right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                      tmp_major_pc, minor_pc, graspable,
                                                                      hand_points, "p_right")

                        if left_points is False and right_points is False:
                            dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                  tmp_major_pc, minor_pc, tmp_grasp_bottom_center])
                            # potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                            #                        tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                if len(dy_potentials) != 0:
                    # Note: we only take the middle grasp from dy direction.
                    potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])

            """ Step3: rotat grasp around an axis(major_pc:green) """
            for dtheta in np.arange(-params['range_dtheta_major'], params['range_dtheta_major'] + 1, params['dtheta']):
                dz_potentials = []
                x, y, z = major_pc
                rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))

                """ Step2: move step by step according to minor_pc """
                for dz in np.arange(-params['num_dz'] * self.gripper.hand_height,
                                    (params['num_dz'] + 1) * self.gripper.hand_height, self.gripper.hand_height):
                    # compute centers and axes
                    tmp_minor_pc = np.dot(rotation, minor_pc)
                    tmp_grasp_normal = np.dot(rotation, normal)
                    tmp_grasp_bottom_center = selected_surface + tmp_minor_pc * dz
                    # go back a bite after rotation dtheta and translation dy!
                    tmp_grasp_bottom_center = self.gripper.init_bite * (-tmp_grasp_normal) + tmp_grasp_bottom_center

                    has_open_points, points_in_open = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                                  major_pc, tmp_minor_pc, graspable,
                                                                                  hand_points, "p_open")
                    bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                   major_pc, tmp_minor_pc, graspable,
                                                                   hand_points, "p_bottom")

                    # grasp_test.append([tmp_grasp_bottom_center, tmp_grasp_normal, major_pc,
                    #                               tmp_minor_pc, tmp_grasp_bottom_center])
                    #
                    # self.show_arrow(tmp_grasp_bottom_center, tmp_grasp_normal, 'r')
                    # self.show_all_grasps(grasp_test)
                    # self.show_points(all_points)
                    # mlab.show()

                    if len(points_in_open) > params['min_points_num'] and bottom_points is False:
                        left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     major_pc, tmp_minor_pc, graspable,
                                                                     hand_points, "p_left")
                        right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                      major_pc, tmp_minor_pc, graspable,
                                                                      hand_points, "p_right")

                        if left_points is False and right_points is False:
                            dz_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal, major_pc,
                                                  tmp_minor_pc, tmp_grasp_bottom_center])
                            # Note: take all grasp from dz direction.
                            potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal, major_pc,
                                                    tmp_minor_pc, tmp_grasp_bottom_center])

                if len(dz_potentials) != 0:
                    # Note: only take the middle grasp from dz direction.
                    potential_grasp.append(dz_potentials[int(np.ceil(len(dz_potentials) / 2) - 1)])

            """ Step4: rotat grasp around an axis(normal:red) """
            for dtheta in np.arange(-params['range_dtheta_normal'], params['range_dtheta_normal'] + 1, params['dtheta']):
                x, y, z = normal
                rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))

                # compute centers and axes
                tmp_normal = -normal
                tmp_major_pc = np.dot(rotation, major_pc)
                tmp_minor_pc = np.dot(rotation, minor_pc)
                # go back a bite after rotation dtheta
                tmp_grasp_bottom_center = self.gripper.init_bite * (-tmp_normal) + selected_surface

                has_open_points, points_in_open = self.check_collision_square(tmp_grasp_bottom_center, tmp_normal,
                                                                              tmp_major_pc, tmp_minor_pc, graspable,
                                                                              hand_points, "p_open")
                bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_normal,
                                                               tmp_major_pc, tmp_minor_pc, graspable,
                                                               hand_points, "p_bottom")

                # grasp_test.append([tmp_grasp_bottom_center, tmp_normal, tmp_major_pc,
                #                                 tmp_minor_pc, tmp_grasp_bottom_center])
                #
                # self.show_arrow(tmp_grasp_bottom_center, tmp_normal, 'r')
                # self.show_all_grasps(grasp_test)
                # self.show_points(all_points)
                # mlab.show()

                if len(points_in_open) > params['min_points_num'] and bottom_points is False:
                    left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_normal,
                                                                 tmp_major_pc, tmp_minor_pc, graspable,
                                                                 hand_points, "p_left")
                    right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_normal,
                                                                  tmp_major_pc, tmp_minor_pc, graspable,
                                                                  hand_points, "p_right")

                    if left_points is False and right_points is False:
                        potential_grasp.append([tmp_grasp_bottom_center, tmp_normal, tmp_major_pc,
                                                tmp_minor_pc, tmp_grasp_bottom_center])
            #             grasp_test.append([tmp_grasp_bottom_center, tmp_normal, tmp_major_pc,
            #                                tmp_minor_pc, tmp_grasp_bottom_center])
            #
            # self.show_arrow(selected_surface, normal, 'r')
            # self.show_all_grasps(grasp_test)
            # self.show_points(all_points)
            # mlab.show()

            """ Step5: approach step by step """
            approach_dist = self.gripper.hand_depth  # use gripper depth
            num_approaches = int(approach_dist / params['approach_step'])
            for ptg in potential_grasp:
                for approach_s in range(num_approaches):
                    # move approach close to the obj
                    tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]
                    tmp_grasp_normal, tmp_major_pc, minor_pc = (ptg[1], ptg[2], ptg[3])

                    is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                    tmp_major_pc, minor_pc, graspable, hand_points)
                    if is_collide:
                        # if collide, go back one step to get a collision free hand position
                        tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step']

                        # final check
                        has_open_points, points_in_open = self.check_collision_square(tmp_grasp_bottom_center,
                                                                                      tmp_grasp_normal,
                                                                                      tmp_major_pc, minor_pc, graspable,
                                                                                      hand_points, "p_open")
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, graspable, hand_points)
                        if len(points_in_open) > params['min_points_num'] and not is_collide:
                            processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                              tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                            if False:
                                logger.info('usefull grasp sample point original: %s', selected_surface)
                                self.show_points(selected_surface, color='r', scale_factor=.005)
                                self.show_grasp_norm_oneside(selected_surface, normal, major_pc,
                                                             minor_pc, scale_factor=0.001)

                        # break after go back one step
                        break

            """ Step6: keep away step by step """
            keepaway_dist = self.gripper.hand_depth/3*2  # use gripper depth
            num_keepaways = int(keepaway_dist / params['keepaway_step'])
            for ptg in potential_grasp:
                for keepaway_s in range(num_keepaways):
                    tmp_grasp_bottom_center = -(ptg[1] * keepaway_s * params['keepaway_step']) + ptg[0]
                    tmp_grasp_normal, tmp_major_pc, minor_pc = (ptg[1], ptg[2], ptg[3])

                    has_open_points, points_in_open = self.check_collision_square(tmp_grasp_bottom_center,
                                                                                  tmp_grasp_normal,
                                                                                  tmp_major_pc, minor_pc, graspable,
                                                                                  hand_points, "p_open")
                    is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                    tmp_major_pc, minor_pc, graspable, hand_points)
                    if len(points_in_open) > params['min_points_num'] and not is_collide:
                        processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                          tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                    # grasp_test.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                    #                   tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                    # if vis:
                    #     logger.info("processed_potential_grasp %d", len(processed_potential_grasp))
                    #     # self.show_all_grasps(processed_potential_grasp)
                    #     self.show_all_grasps(grasp_test)
                    #     self.show_points(all_points)
                    #     self.display_grasps3d(grasps, 'g')
                    #     mlab.show()

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)

        # convert grasps to dexnet formate
        for grasp in processed_potential_grasp:
            grasp_bottom_center = np.array(grasp[0])
            grasp_normal = np.array(grasp[1])
            major_pc = np.array(grasp[2])
            minor_pc = np.array(grasp[3])

            grasp_top_center = grasp_bottom_center + self.gripper.hand_depth * grasp_normal
            grasp3d = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
                grasp_top_center, major_pc, self.gripper.hand_outer_diameter-self.gripper.finger_width,
                depth=self.gripper.hand_depth, normal=grasp_normal, minor_pc=minor_pc), type='frame')
            grasps.append(grasp3d)

        if False:
            logger.info("generate potential grasp %d", len(processed_potential_grasp))
            self.show_all_grasps(processed_potential_grasp)
            # self.show_all_grasps(grasp_test)
            self.show_points(all_points)
            # self.display_grasps3d(grasps, 'g')
            mlab.show()

        # return grasps
        logger.info("generate %d grasps, took %.2f s", len(grasps), time.perf_counter()-start)
        return grasps
