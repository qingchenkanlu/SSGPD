# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import scipy.stats as stats
import pcl
import dexnet

from dexnet.grasping import Grasp, Contact3D, ParallelJawPtGrasp3D, PointGraspMetrics3D  # , GraspableObject3D
from autolab_core import RigidTransform
import scipy
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import mayavi.mlab as mlab
except:
    try:
        import mayavi.mlab as mlab
    except ImportError:
        mlab = []
        logging.error('Failed to import mayavi')


# class GraspSampler(metaclass=ABCMeta):
class GraspSampler:
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

    # NOTE: Unused
    def generate_grasps_stable_poses(self, graspable, stable_poses, target_num_grasps=None, grasp_gen_mult=5,
                                     max_iter=3, sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object, aligning the approach angles to the object stable poses.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            list of stable poses for the object with ids read from the database
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        vis : bool
        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # sample dense grasps
        unaligned_grasps = self.generate_grasps(graspable, target_num_grasps=target_num_grasps,
                                                grasp_gen_mult=grasp_gen_mult,
                                                max_iter=max_iter, vis=vis)

        # align for each stable pose
        grasps = {}
        print(sample_approach_angles)  # add by Liang
        for stable_pose in stable_poses:
            grasps[stable_pose.id] = []
            for grasp in unaligned_grasps:
                aligned_grasp = grasp.perpendicular_table(grasp)
                grasps[stable_pose.id].append(copy.deepcopy(aligned_grasp))
        return grasps

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

    def show_points(self, point, color='lb', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

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

    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_points(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        # mlab.points3d(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], color=(0, 0, 1), scale_factor=0.005)
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles, color=color, opacity=0.8)

    def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
                               minor_pc, graspable, p, way, vis=False):
        normal = approach_normal
        major = binormal
        minor = minor_pc

        approach_normal = approach_normal.reshape(1, 3)
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        binormal = binormal.reshape(1, 3)
        binormal = binormal / np.linalg.norm(binormal)
        minor_pc = minor_pc.reshape(1, 3)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)
        matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
        grasp_matrix = matrix.T  # same as cal the inverse
        if isinstance(graspable, dexnet.grasping.graspable_object.GraspableObject3D):
            points = graspable.sdf.surface_points(grid_basis=False)[0]
        else:
            points = graspable
        points = points - grasp_bottom_center.reshape(1, 3)
        # points_g = points @ grasp_matrix
        tmp = np.dot(grasp_matrix, points.T)
        points_g = tmp.T
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
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

        if vis:
            print("points_in_area", way, len(points_in_area))
            mlab.figure(bgcolor=(1, 1, 1), size=(640, 480))
            mlab.clf()
            # self.show_one_point(np.array([0, 0, 0]))
            self.show_grasp_norm_oneside(grasp_bottom_center, normal, major,
                                         minor, scale_factor=0.001)
            self.show_grasp_3d(p)
            self.show_points(points_g)
            if len(points_in_area) != 0:
                self.show_points(points_g[points_in_area], color='r')
            # mlab.show()

        # print("points_in_area", way, len(points_in_area))
        return has_p, points_in_area

    def show_all_grasps(self, grasps_for_show):

        for grasp_ in grasps_for_show:
            grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
            approach_normal = grasp_[1]
            binormal = grasp_[2]
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, binormal)
            self.show_grasp_3d(hand_points)

    def show_surface_points(self, graspable):
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        self.show_points(surface_points)

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

    def cal_surface_property(self, graspable, selected_surface, r_ball,
                             point_amount, max_trial, vis=False):
        tmp_count = 0
        M = np.zeros((3, 3))
        trial = 0
        old_normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(selected_surface))
        if old_normal is None:
            logger.warning("The selected point has no norm according to meshpy!")
            return None
        while tmp_count < point_amount and trial < max_trial:
            trial += 1
            neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball
            normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(neighbor))
            if normal is None:
                continue
            normal = normal.reshape(-1, 1)
            if np.linalg.norm(normal) != 0:
                normal /= np.linalg.norm(normal)
            if vis:
                # show the r-ball performance
                neighbor = neighbor.reshape(-1, 1)
                # self.show_line(neighbor, normal * 0.05 + neighbor)
            M += np.matmul(normal, normal.T)
            tmp_count = tmp_count + 1

        if trial == max_trial:
            logger.warning("rball computation failed over %d", max_trial)
            return None

        eigval, eigvec = np.linalg.eig(M)  # compared computed normal
        minor_pc = eigvec[np.argmin(eigval)]  # minor principal curvature
        minor_pc /= np.linalg.norm(minor_pc)
        new_normal = eigvec[np.argmax(eigval)]  # estimated surface normal
        new_normal /= np.linalg.norm(new_normal)
        major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
        if np.linalg.norm(major_pc) != 0:
            major_pc = major_pc / np.linalg.norm(major_pc)
        return old_normal, new_normal, major_pc, minor_pc

    def display_grasps3d(self, grasps, color):
        """
        display ParrallelJawPtGrasp3D
        """
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
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

    def show(self):
        mlab.show()


class AntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone,
    then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """

    def sample_from_cone(self, n, tx, ty, num_samples=1):
        """ Samples directoins from within the friction cone using uniform sampling.

        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        num_samples : int
            number of directions to sample

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
       """
        v_samples = []
        for i in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = self.friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Checks whether or not a direction is in the friction cone.
        This is equivalent to whether a grasp will slip using a point contact model.

        Parameters
        ----------
        cone : 3xN :obj:`numpy.ndarray`
            supporting vectors of the friction cone
        n : 3x1 :obj:`numpy.ndarray`
            outward pointing surface normal vector at c1
        v : 3x1 :obj:`numpy.ndarray`
            direction vector

        Returns
        -------
        in_cone : bool
            True if alpha is within the cone
        alpha : float
            the angle between the normal and v
        """
        if (v.dot(cone) < 0).any():  # v should point in same direction as cone
            v = -v  # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point 均匀随机扰动"""
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def sample_grasps(self, graspable, num_grasps, vis=False, **kwargs):
        """Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample NOTE: Unused in AntipodalGraspSampler!
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        print("[INFO] sample grasps antipodal.")

        # get surface points 获取表面点
        grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
        print("shuffled surface points num:", len(shuffled_surface_points))

        for k, x_surf in enumerate(shuffled_surface_points):  # 遍历各个采样点
            print("current point:", k, "generated grasps:", len(grasps))
            start_time = time.clock()

            # perturb grasp for num samples 调整当前采样的抓取姿态
            for i in range(self.num_samples):  # 在当前采样点处生成num_samples个抓取姿态
                # perturb contact
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)  # 给点添加随机扰动

                # compute friction cone faces 计算摩擦锥面
                c1 = Contact3D(graspable, x1, in_direction=None)  # 3D接触点
                _, tx1, ty1 = c1.tangents()  # 计算表面法线和切线
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)  # 计算摩擦锥
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(n1, tx1, ty1, num_samples=1)  # 在摩擦锥内均匀采样一个方向向量
                sample_time = time.clock()

                for v in v_samples:  # 处理各个方向向量, 这里仅生成了一个
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)  # 接触点 (3*1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=False)  # 摩擦锥(3*8)
                        plt.clf()
                        plt.gcf()
                        # plt.ion()  # 使用交互模式, 不支持多进程!
                        ax = plt.gca(projection='3d')
                        for j in range(cone1.shape[1]):
                            # print(x1_grid[0] - cone1_grid[0])
                            # print(x1_grid[1] - cone1_grid[1])
                            # print("cone1_grid[0]", cone1_grid[0], cone1_grid[0].shape)
                            # print("cone1_grid[1]", cone1_grid[1], cone1_grid[1].shape)
                            # ax.scatter(cone1_grid[0], cone1_grid[1], cone1_grid[2], s=50,  c='m')

                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1],
                                       x1_grid[2] - cone1_grid[2], s=50, c='m')

                    # random axis flips since we don't have guarantees on surface normal directoins
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts 利用单个接触点生成平行夹抓的抓取姿态
                    grasp, c1, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(
                        graspable, x1, v, self.gripper.max_width,
                        min_grasp_width_world=self.gripper.min_width, vis=False)

                    if grasp is None or c2 is None:
                        continue

                    self.show_points(surface_points)
                    self.show_points(grasp.center, color='r', scale_factor=.005)
                    self.show_points(c1.point, color='g', scale_factor=.005)
                    self.show_points(c2.point, color='g', scale_factor=.005)
                    self.display_grasps3d([grasp], 'b')
                    mlab.show()

                    if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:  # NOTE:随机调整approach角度
                        # print('[INFO] random approach angle')
                        angle_candidates = np.arange(-90, 91, 30)  # [-90,90]
                        np.random.shuffle(angle_candidates)
                        for grasp_angle in angle_candidates:
                            grasp.approach_angle_ = grasp_angle
                            # get true contacts (previous is subject to variation)
                            success, c = grasp.close_fingers(graspable, vis=vis)
                            if not success:
                                # print("not success:", grasp_angle)
                                continue
                            else:
                                # print("success grasp angle:", grasp_angle)
                                pass
                            break
                        else:
                            continue
                    else:
                        print("angle:", grasp.approach_angle_)
                        success, c = grasp.close_fingers(graspable, vis=vis)
                        if not success:
                            continue
                    c1 = c[0]
                    c2 = c[1]

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2  # 计算第二接触点摩擦锥
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=30)
                        print("show")
                        plt.show(block=False)
                        # plt.savefig("filename.png")
                        # exit("debug")
                        time.sleep(0.5)
                        plt.close()  # lol

                    # check friction cone
                    if PointGraspMetrics3D.force_closure(c1, c2, self.friction_coef):  # NOTE: 力闭合检查
                        grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        print("[INFO] sample grasps onece.")
        return grasps


class GpgGraspSampler(GraspSampler):
    """
    Sample grasps by GPG.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=30, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
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
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 5,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.01,
            'keepaway_step': 0.01,
            'max_trail_for_r_ball': 3000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        all_points = surface_points
        # construct pynt point cloud and voxel grid
        p_cloud = pcl.PointCloud(surface_points.astype(np.float32))
        voxel = p_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * params['voxel_grid_ratio']] * 3))
        surface_points = voxel.filter().to_array()

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []
        grasp_test = []

        # # grasp_test.append([tmp_grasp_bottom_center, tmp_grasp_normal,
        # #                    # [-0.00393718 -0.10461882  0.17252018]  [-0.18408447  0.98253861 -0.02703294]
        # #                    tmp_major_pc, minor_pc, tmp_grasp_bottom_center])
        # grasp_te=([np.array([-0.00393718, -0.10461882,  0.17252018]), np.array([-0.18408447,  0.98253861, -0.02703294]),
        #                    #  [-0.00393718 -0.10461882  0.17252018]  [-0.18408447  0.98253861 -0.02703294]
        #                    np.array([-0.98289787, -0.18387343,  0.01011668]), np.array([0.00496939, 0.02843294, 0.99958335]),
        #                    np.array([-0.00393718, -0.10461882,  0.17252018])])  # [-0.98289787 -0.18387343  0.01011668]  [0.00496939 0.02843294 0.99958335]
        #
        # for i in range(1):
        #     # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
        #     #     # tmp_grasp_bottom_center + self.gripper.hand_depth*tmp_grasp_normal,
        #     #     [-0.00393718, -0.10461882,  0.17252018],  #
        #     #     [-0.98289787, -0.18387343,  0.01011668],  #
        #     #     self.gripper.max_width, angle=10*i, normal=[-0.18408447,  0.98253861, -0.02703294], minor_pc=[0.00496939, 0.02843294, 0.99958335]), type='frame')  # 0.085
        #     # print(g.unrotated_full_axis)
        #     g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
        #         # tmp_grasp_bottom_center + self.gripper.hand_depth*tmp_grasp_normal,
        #         [-0.00393718, -0.10461882, 0.17252018],  #
        #         [-0.98289787, -0.18387343, 0.01011668],  #
        #         self.gripper.max_width, angle=10 * i))  # 0.085
        #     print(g.unrotated_full_axis)
        #     # self.show_points(np.array([-0.00393718, -0.10461882,  0.17252018]), 'r', 0.005)
        #     # print(10*i)
        #     # self.show_points(all_points)
        #     # self.show_all_grasps([grasp_te])
        #     # self.display_grasps3d([g], 'g')
        #     # mlab.quiver3d(-0.00393718, -0.10461882, 0.17252018, -0.98289787, -0.18387343, 0.01011668,
        #     #               scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')
        #     # mlab.show()
        #     pass


        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            # cal major principal curvature
            # r_ball = max(self.gripper.hand_depth, self.gripper.hand_outer_diameter)
            r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            point_amount = params['num_rball_points']
            max_trial = params['max_trail_for_r_ball']

            # NOTE: we can not directly sample from point clouds so we use a relatively small radius.
            """ cal local frame: normal, major_pc, minor_pc """
            ret = self.cal_surface_property(graspable, selected_surface, r_ball,
                                            point_amount, max_trial, vis=vis)
            if ret is None:
                continue
            else:
                old_normal, new_normal, major_pc, minor_pc = ret

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(old_normal, new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1.]:  # NOTE: here we can now know the direction of norm, outward
                """ Step1: rotat grasp around an axis(minor_pc:blue) """
                potential_grasp = []
                for dtheta in np.arange(-params['range_dtheta'], params['range_dtheta'] + 1, params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))

                    """ Step2: move step by step according to major_pc """
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width, self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.gripper.init_bite * (-tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, graspable,
                                                                       hand_points, "p_bottom")

                        if open_points is True and bottom_points is False:
                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, graspable,
                                                                          hand_points, "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                                # potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                #                        tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                    if len(dy_potentials) != 0:
                        # FIXME: we only take the middle grasp from dy direction.
                        potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])

                """ Step3: approach step by step """
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
                            open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, graspable, hand_points)
                            if open_points and not is_collide:
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                                if vis:
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.show_points(selected_surface, color='r', scale_factor=.005)
                                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir,
                                                                 major_pc * normal_dir,
                                                                 minor_pc, scale_factor=0.001)

                            # break after go back one step
                            break

                """ Step4: keep away step by step """
                keepaway_dist = self.gripper.hand_depth/3*2  # use gripper depth
                num_keepaways = int(keepaway_dist / params['keepaway_step'])
                for ptg in potential_grasp:
                    for keepaway_s in range(num_keepaways):
                        tmp_grasp_bottom_center = -(ptg[1] * keepaway_s * params['keepaway_step']) + ptg[0]
                        tmp_grasp_normal, tmp_major_pc, minor_pc = (ptg[1], ptg[2], ptg[3])

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                     tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, graspable, hand_points)
                        if open_points and not is_collide:
                            processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                              tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                        grasp_test.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                          tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                        # if vis:
                        #     logger.info("processed_potential_grasp %d", len(processed_potential_grasp))
                        #     # self.show_all_grasps(processed_potential_grasp)
                        #     self.show_all_grasps(grasp_test)
                        #     self.show_points(all_points)
                        #     self.display_grasps3d(grasps, 'g')
                        #     mlab.show()

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)

        for grasp in processed_potential_grasp:
            grasp_bottom_center = np.array(grasp[0])
            grasp_normal = np.array(grasp[1])
            major_pc = np.array(grasp[2])
            minor_pc = np.array(grasp[3])

            grasp_top_center = grasp_bottom_center + self.gripper.hand_depth * grasp_normal
            grasp3d = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
                grasp_top_center, major_pc, self.gripper.max_width, normal=grasp_normal,
                minor_pc=minor_pc), type='frame')
            grasps.append(grasp3d)

        if vis:
            logger.info("processed_potential_grasp %d", len(processed_potential_grasp))
            # self.show_all_grasps(processed_potential_grasp)
            self.show_all_grasps(grasp_test)
            self.show_points(all_points)
            self.display_grasps3d(grasps, 'g')
            mlab.show()

        # return processed_potential_grasp
        return grasps


class UniformGraspSampler(GraspSampler):
    """ Sample grasps by sampling pairs of points on the object surface uniformly at random.
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
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
        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]
        i = 0
        grasps = []

        # get all grasps
        while len(grasps) < num_grasps and i < max_num_samples:
            # get candidate contacts
            indices = np.random.choice(num_surface, size=2, replace=False)
            c0 = surface_points[indices[0], :]
            c1 = surface_points[indices[1], :]

            gripper_distance = np.linalg.norm(c1 - c0)
            if self.gripper.min_width < gripper_distance < self.gripper.max_width:
                # compute centers and axes
                grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c0, c1)
                grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c0, c1)
                g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center,
                                                                                        grasp_axis,
                                                                                        self.gripper.max_width))
                # keep grasps if the fingers close
                if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                    angle_candidates = np.arange(-90, 120, 30)
                    np.random.shuffle(angle_candidates)
                    for grasp_angle in angle_candidates:
                        g.approach_angle_ = grasp_angle
                        # get true contacts (previous is subject to variation)
                        success, contacts = g.close_fingers(graspable, vis=vis)
                        if not success:
                            continue
                        break
                    else:
                        continue
                else:
                    success, contacts = g.close_fingers(graspable, vis=vis)

                if success:
                    grasps.append(g)
            i += 1

        return grasps


class GaussianGraspSampler(GraspSampler):
    """ Sample grasps by sampling a center from a gaussian with mean at the object center of mass
    and grasp axis by sampling the spherical angles uniformly at random.
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, sigma_scale=2.5, **kwargs):
        """
        Returns a list of candidate grasps for graspable object by Gaussian with
        variance specified by principal dimensions.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        sigma_scale : float
            the number of sigmas on the tails of the Gaussian for each dimension
        vis : bool
            visualization

        Returns
        -------
        :obj:`list` of obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get object principal axes
        center_of_mass = graspable.mesh.center_of_mass
        principal_dims = graspable.mesh.principal_dims()
        sigma_dims = principal_dims / (2 * sigma_scale)

        # sample centers
        grasp_centers = stats.multivariate_normal.rvs(
            mean=center_of_mass, cov=sigma_dims ** 2, size=num_grasps)

        # samples angles uniformly from sphere
        u = stats.uniform.rvs(size=num_grasps)
        v = stats.uniform.rvs(size=num_grasps)
        thetas = 2 * np.pi * u
        phis = np.arccos(2 * v - 1.0)
        grasp_dirs = np.array([np.sin(phis) * np.cos(thetas), np.sin(phis) * np.sin(thetas), np.cos(phis)])
        grasp_dirs = grasp_dirs.T

        # convert to grasp objects
        grasps = []
        for i in range(num_grasps):
            grasp = ParallelJawPtGrasp3D(
                ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i, :], grasp_dirs[i, :],
                                                               self.gripper.max_width))

            if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                angle_candidates = np.arange(-90, 120, 30)
                np.random.shuffle(angle_candidates)
                for grasp_angle in angle_candidates:
                    grasp.approach_angle_ = grasp_angle
                    # get true contacts (previous is subject to variation)
                    success, contacts = grasp.close_fingers(graspable, vis=vis)
                    if not success:
                        continue
                    break
                else:
                    continue
            else:
                success, contacts = grasp.close_fingers(graspable, vis=vis)

            # add grasp if it has valid contacts
            if success and np.linalg.norm(contacts[0].point - contacts[1].point) > self.min_contact_dist:
                grasps.append(grasp)

        # visualize
        if vis:
            for grasp in grasps:
                plt.clf()
                plt.gcf()
                plt.ion()
                grasp.close_fingers(graspable, vis=vis)
                plt.show(block=False)
                time.sleep(0.5)

            grasp_centers_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_centers.T)
            grasp_centers_grid = grasp_centers_grid.T
            com_grid = graspable.sdf.transform_pt_obj_to_grid(center_of_mass)

            plt.clf()
            ax = plt.gca(projection='3d')
            # graspable.sdf.scatter()
            ax.scatter(grasp_centers_grid[:, 0], grasp_centers_grid[:, 1], grasp_centers_grid[:, 2], s=60, c='m')
            ax.scatter(com_grid[0], com_grid[1], com_grid[2], s=120, c='y')
            ax.set_xlim3d(0, graspable.sdf.dims_[0])
            ax.set_ylim3d(0, graspable.sdf.dims_[1])
            ax.set_zlim3d(0, graspable.sdf.dims_[2])
            plt.show()

        return grasps


class PointGraspSampler(GraspSampler):
    """
    Sample grasps by PointGraspSampler
    TODO: since gpg sampler changed a lot, this class need to totally rewrite
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
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
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 3000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        all_points = surface_points
        # construct pynt point cloud and voxel grid
        p_cloud = pcl.PointCloud(surface_points.astype(np.float32))
        voxel = p_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * params['voxel_grid_ratio']] * 3))
        surface_points = voxel.filter().to_array()

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            # begin of modification 5: Gaussian over height, select more point in the middle
            # we can use the top part of the point clouds to generate more sample points
            min_height = min(surface_points[:, 2])
            max_height = max(surface_points[:, 2])

            selected_height = min_height + np.random.normal(3 * (max_height - min_height) / 4,
                                                            (max_height - min_height) / 6)
            ind_10 = np.argsort(abs(surface_points[:, 2] - selected_height))[:10]
            ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5
            # ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            # cal major principal curvature
            r_ball = max(self.gripper.hand_depth, self.gripper.hand_outer_diameter)
            point_amount = params['num_rball_points']
            max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.
            ret = self.cal_surface_property(graspable, selected_surface, r_ball,
                                            point_amount, max_trial, vis=params['debug_vis'])
            if ret is None:
                continue
            else:
                old_normal, new_normal, major_pc, minor_pc = ret

            for normal_dir in [-1., 1.]:  # FIXME: here we do not know the direction of the object normal
                grasp_bottom_center = self.gripper.init_bite * new_normal * -normal_dir + selected_surface
                new_normal = normal_dir * new_normal
                major_pc = normal_dir * major_pc

                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal, major_pc,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)

                # some magic number referred from origin paper
                potential_grasp = []
                extra_potential_grasp = []

                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc)
                        tmp_grasp_normal = np.dot(rotation, new_normal)
                        tmp_grasp_bottom_center = grasp_bottom_center + tmp_major_pc * dy

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, graspable,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, graspable,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        # potential_grasp += dy_potentials
                        potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])

                    # get more potential_grasp by moving along minor_pc
                    if len(potential_grasp) != 0:
                        self.show_points(selected_surface, color='r', scale_factor=.005)
                        for pt in potential_grasp:
                            for dz in range(-5, 5):
                                new_center = minor_pc * dz * 0.01 + pt[0]
                                extra_potential_grasp.append([new_center, pt[1], pt[2], pt[3]])
                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist // params['approach_step'])
                for ptg in extra_potential_grasp:
                    for _ in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        not_collide = self.check_approach_grasp(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points)

                        if not not_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center = -ptg[1] * params['approach_step'] + ptg[0]
                            # final check
                            open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_open")
                            not_collide = self.check_approach_grasp(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                    tmp_major_pc, minor_pc, graspable, hand_points)
                            if open_points and not_collide:
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc])

                                self.show_points(selected_surface, color='r', scale_factor=.005)
                                if params['debug_vis']:
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points,
                                                                "p_open", vis=True)
                        break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            if not sampled_surface_amount % 60:  # params['debug_vis']:
                self.show_all_grasps(all_points, processed_potential_grasp)
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class OldPointGraspSampler(GraspSampler):
    """
    Sample grasps by PointGraspSampler
    """

    def show_obj(self, graspable, color='b', clear=False):
        if clear:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()

        # plot the obj
        ax = plt.gca(projection='3d')
        surface = graspable.sdf.surface_points()[0]
        surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                   s=np.ones_like(surface[:, 0]) * 0.3, c=color)

    def show_grasp_norm(self, graspable, grasp_center, grasp_bottom_center,
                        grasp_normal, grasp_axis, minor_pc, color='b', clear=False):
        if clear:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()

        ax = plt.gca(projection='3d')
        grasp_center_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_center)
        ax.scatter(grasp_center_grid[0], grasp_center_grid[1], grasp_center_grid[2], marker='s', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * grasp_axis * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='^', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * minor_pc * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='^', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='*', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * grasp_normal * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='*', c=color)

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
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
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 0.3,
            'range_dtheta': 0.30,
            'max_chain_length': 20,
            'max_retry_times': 100
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]

        i = 0
        self.grasps = []

        # ____count = 0
        # get all grasps
        while len(self.grasps) < num_grasps and i < max_num_samples:
            # print('sample times:', ____count)
            # ____count += 1
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(surface_points[:, 2])
            # max_height = max(surface_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(surface_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]

            # end of modification 5
            ind = np.random.choice(num_surface, size=1, replace=False)
            grasp_bottom_center = surface_points[ind, :]
            grasp_bottom_center = grasp_bottom_center.reshape(3)

            for ind in range(params['max_chain_length']):
                # if not graspable.sdf.on_surface(graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center))[0]:
                #     print('first damn it!')
                #     from IPython import embed; embed()
                new_grasp_bottom_center = self.sample_chain(grasp_bottom_center, graspable,
                                                            params, vis)
                if new_grasp_bottom_center is None:
                    i += ind
                    break
                else:
                    grasp_bottom_center = new_grasp_bottom_center
            else:
                i += params['max_chain_length']
            print('Chain broken, length:', ind, 'amount:', len(self.grasps))
        return self.grasps

    def sample_chain(self, grasp_bottom_center, graspable, params, vis):
        grasp_success = False
        grasp_normal = graspable.sdf.surface_normal(
            graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center))
        for normal_dir in [-1., 1.]:  # FIXME: here we assume normal is outward
            grasp_center = self.gripper.max_depth * normal_dir * grasp_normal + grasp_bottom_center
            r_ball = max(self.gripper.max_depth, self.gripper.max_width)
            # cal major principal curvature
            tmp_count = 0
            M = np.zeros((3, 3))
            while tmp_count < params['num_rball_points']:
                neighbor = grasp_bottom_center + 2 * (np.random.rand(3) - 0.5) * r_ball
                normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(neighbor))
                if normal is None:
                    continue
                normal = normal.reshape(-1, 1)
                M += np.matmul(normal, normal.T)
                tmp_count = tmp_count + 1
            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[np.argmin(eigval)]  # minor principal curvature
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[np.argmax(eigval)]  # estimated surface normal
            new_normal /= np.linalg.norm(new_normal)
            # major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            # FIXME: We can not get accurate normal, so we use grasp_normal instead
            major_pc = np.cross(minor_pc, grasp_normal)
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)
            grasp_axis = major_pc

            g = ParallelJawPtGrasp3D(
                ParallelJawPtGrasp3D.configuration_from_params(
                    grasp_center,
                    grasp_axis,
                    self.gripper.max_width))
            grasp_success, _ = g.close_fingers(graspable, vis=vis)
            if grasp_success:
                self.grasps.append(g)
        if not grasp_success:
            return None

        trial = 0
        next_grasp_bottom_center = None
        while trial < params['max_retry_times'] and next_grasp_bottom_center is None:
            trial += 1
            dy = np.random.uniform(-params['num_dy'] * self.gripper.finger_width,
                                   (params['num_dy']) * self.gripper.finger_width)
            dtheta = np.random.uniform(-params['range_dtheta'], params['range_dtheta'])

            for tmp_normal_dir in [-1., 1.]:
                # get new grasp sample from a chain
                # some magic number referred from origin paper

                # compute centers and axes
                x, y, z = minor_pc
                rotation = RigidTransform.rotation_from_quaternion(
                    np.array([dtheta / 180 * np.pi, x, y, z]))
                tmp_grasp_axis = np.dot(rotation, grasp_axis)
                tmp_grasp_normal = np.dot(rotation, grasp_normal)

                tmp_grasp_bottom_center = grasp_bottom_center + tmp_grasp_axis * dy
                # TODO: find contact
                # FIXME: 0.2 is the same as close_finger()
                approach_dist = 0.2
                approach_dist_grid = graspable.sdf.transform_pt_obj_to_grid(approach_dist)
                num_approach_samples = int(Grasp.samples_per_grid * approach_dist_grid / 2)
                approach_loa = ParallelJawPtGrasp3D.create_line_of_action(tmp_grasp_bottom_center,
                                                                          -tmp_grasp_normal * tmp_normal_dir,
                                                                          approach_dist,
                                                                          graspable,
                                                                          num_approach_samples,
                                                                          min_width=0)
                contact_found, contact = ParallelJawPtGrasp3D.find_contact(approach_loa,
                                                                           graspable, vis=vis)
                if not contact_found:
                    continue
                else:
                    if not graspable.sdf.on_surface(graspable.sdf.transform_pt_obj_to_grid(contact.point))[0]:
                        # print('damn it!')
                        pass
                    else:
                        next_grasp_bottom_center = contact.point
                        break
        print('amount:', len(self.grasps), 'next center:', next_grasp_bottom_center)
        return next_grasp_bottom_center


class GpgGraspSamplerPcl(GraspSampler):
    """
    Sample grasps by GPG with pcl directly.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, point_cloud, points_for_sample, all_normal, num_grasps=20, max_num_samples=200,
                      show_final_grasp=False, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        point_cloud :
        all_normal :
        num_grasps : int
            the number of grasps to generate

        show_final_grasp :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 1000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        all_points = point_cloud.to_array()
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(all_points[:, 2])
            # max_height = max(all_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(all_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5

            # for ros, we needed to judge if the robot is at HOME

            scipy.random.seed()  # important! without this, the worker will get a pseudo-random sequences.
            ind = np.random.choice(points_for_sample.shape[0], size=1, replace=False)
            selected_surface = points_for_sample[ind, :].reshape(3, )
            if show_final_grasp:
                mlab.points3d(selected_surface[0], selected_surface[1], selected_surface[2],
                              color=(1, 0, 0), scale_factor=0.005)

            # cal major principal curvature 计算主曲率
            # r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            r_ball = max(self.gripper.hand_outer_diameter - self.gripper.finger_width, self.gripper.hand_depth,
                         self.gripper.hand_height / 2.0)
            # point_amount = params['num_rball_points']
            # max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.

            M = np.zeros((3, 3))

            # neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball

            selected_surface_pc = pcl.PointCloud(selected_surface.reshape(1, 3))
            kd = point_cloud.make_kdtree_flann()
            kd_indices, sqr_distances = kd.radius_search_for_cloud(selected_surface_pc, r_ball, 100)
            for _ in range(len(kd_indices[0])):
                if sqr_distances[0, _] != 0:
                    # neighbor = point_cloud[kd_indices]
                    normal = all_normal[kd_indices[0, _]]
                    normal = normal.reshape(-1, 1)
                    if np.linalg.norm(normal) != 0:
                        normal /= np.linalg.norm(normal)
                    M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour")
                print("Here is a bug, if points amount is too little it will keep trying and never go outside.")
                continue
            else:
                logger.info("Selected a good sample point.")

            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)
            major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(all_normal[ind], new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1]:  # FIXED: we know the direction of norm is outward as we know the camera pos
                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    self.show_points(all_points)
                    # show real norm direction: if new_norm has very diff than pcl cal norm, then maybe a bug.
                    self.show_line(selected_surface, (selected_surface + all_normal[ind]*0.05).reshape(3))
                    mlab.show()

                # some magic number referred from origin paper
                potential_grasp = []
                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    dtheta = np.float64(dtheta)
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.gripper.init_bite * (
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, all_points,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, all_points,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, all_points,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])

                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        center_dy = dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)]
                        # we check if the gripper has a potential to collide with the table
                        # by check if the gripper is grasp from a down to top direction
                        finger_top_pos = center_dy[0] + center_dy[1] * self.gripper.hand_depth
                        # [- self.gripper.hand_depth * 0.5] means we grasp objects as a angel larger than 30 degree
                        if finger_top_pos[2] < center_dy[0][2] - self.gripper.hand_depth * 0.5:
                            potential_grasp.append(center_dy)

                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist / params['approach_step'])

                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, point_cloud, hand_points)

                        if is_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step'] * 3
                            # minus 3 means we want the grasp go back a little bitte more.

                            # here we check if the gripper collide with the table.
                            hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                                tmp_grasp_normal,
                                                                tmp_major_pc)[1:]
                            min_finger_end = hand_points_[:, 2].min()
                            min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]

                            safety_dis_above_table = 0.01
                            if min_finger_end < safety_dis_above_table:
                                min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                                x = -min_finger_pos[2]*tmp_grasp_normal[0]/tmp_grasp_normal[2]+min_finger_pos[0]
                                y = -min_finger_pos[2]*tmp_grasp_normal[1]/tmp_grasp_normal[2]+min_finger_pos[1]
                                p_table = np.array([x, y, 0])  # the point that on the table
                                dis_go_back = np.linalg.norm([min_finger_pos, p_table]) + safety_dis_above_table
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back
                            else:
                                # if the grasp is not collide with the table, do not change the grasp
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                            # final check
                            _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, all_points, hand_points)
                            if (len(open_points) > 10) and not is_collide:
                                # here 10 set the minimal points in a grasp, we can set a parameter later
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc,
                                                                  tmp_grasp_bottom_center_modify])
                                if params['debug_vis']:
                                    self.show_points(selected_surface, color='r', scale_factor=.005)
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, all_points, hand_points,
                                                                "p_open", vis=True)
                                break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            print("current amount of sampled surface:", sampled_surface_amount)
            if params['debug_vis']:  # not sampled_surface_amount % 5:
                if len(all_points) > 10000:
                    pc = pcl.PointCloud(all_points)
                    voxel = pc.make_voxel_grid_filter()
                    voxel.set_leaf_size(0.01, 0.01, 0.01)
                    point_cloud = voxel.filter()
                    all_points = point_cloud.to_array()
                self.show_all_grasps(all_points, processed_potential_grasp)
                self.show_points(all_points, scale_factor=0.008)
                mlab.show()
            print("The grasps number got by modified GPG:", len(processed_potential_grasp))
            if len(processed_potential_grasp) >= num_grasps or sampled_surface_amount >= max_num_samples:
                if show_final_grasp:
                    self.show_all_grasps(all_points, processed_potential_grasp)
                    self.show_points(all_points, scale_factor=0.002)
                    mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
                    table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
                    triangles = [(1, 2, 3), (0, 1, 3)]
                    mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
                                         triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
                    mlab.show()
                return processed_potential_grasp
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class GpgGraspSamplerPclPcd(GraspSampler):
    """
    Sample grasps by GPG with pcl directly.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, point_cloud, points_for_sample, all_normal, num_grasps=20, max_num_samples=200,
                      show_final_grasp=False, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        point_cloud :
        all_normal :
        num_grasps : int
            the number of grasps to generate

        show_final_grasp :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 3,  # number
            'dtheta': 5,  # unit degree
            'range_dtheta': 30,
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 1000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        all_points = point_cloud.to_array()
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(all_points[:, 2])
            # max_height = max(all_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(all_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5

            scipy.random.seed()  # important! without this, the worker will get a pseudo-random sequences.
            ind = np.random.choice(points_for_sample.shape[0], size=1, replace=False)
            selected_surface = points_for_sample[ind, :].reshape(3, )
            if show_final_grasp:
                mlab.points3d(selected_surface[0], selected_surface[1], selected_surface[2],
                              color=(1, 0, 0), scale_factor=0.005)

            # cal major principal curvature 计算主曲率
            # r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            r_ball = max(self.gripper.hand_outer_diameter - self.gripper.finger_width, self.gripper.hand_depth,
                         self.gripper.hand_height / 2.0)
            # point_amount = params['num_rball_points']
            # max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.

            M = np.zeros((3, 3))

            # neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball

            # 通过KdTree获取采样点
            selected_surface_pc = pcl.PointCloud(selected_surface.reshape(1, 3))
            kd = point_cloud.make_kdtree_flann()
            kd_indices, sqr_distances = kd.radius_search_for_cloud(selected_surface_pc, r_ball, 100)
            for _ in range(len(kd_indices[0])):
                if sqr_distances[0, _] != 0:
                    # neighbor = point_cloud[kd_indices]
                    normal = all_normal[kd_indices[0, _]]
                    normal = normal.reshape(-1, 1)
                    if np.linalg.norm(normal) != 0:
                        normal /= np.linalg.norm(normal)
                    M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour")
                print("Here is a bug, if points amount is too little it will keep trying and never go outside.")
                continue
            else:
                logger.info("Selected a good sample point.")

            # NOTE: 坐标系定义
            #  normal(法线r-x)：对应gpd的approach
            #  major_pc(主曲率g-y)：对应gpd的axis
            #  minor_pc(次曲率b-z)：对应gpd的binormal
            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)  # major principal curvature 次曲率
            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)
            major_pc = np.cross(minor_pc, new_normal)  # major principal curvature 主曲率
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(all_normal[ind], new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1]:  # FIXED: we know the direction of norm is outward as we know the camera pos
                if params['debug_vis']:
                    # example of show grasp frame
                    print("\033[0;32m%s\033[0m" % "[Debug] Show grasp frame.")
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    self.show_points(all_points)
                    # show real norm direction: if new_norm has very diff than pcl cal norm, then maybe a bug.
                    self.show_line(selected_surface, (selected_surface + all_normal[ind]*0.05).reshape(3))
                    mlab.show()

                # 采样点处抓取姿态调整
                # some magic number referred from origin paper
                potential_grasp = []
                for dtheta in np.arange(-params['range_dtheta'], params['range_dtheta'] + 1, params['dtheta']):  # 调整角度
                    dy_potentials = []
                    x, y, z = minor_pc  # 次曲率
                    dtheta = np.float64(dtheta)
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,  # 次曲率(y)方向平移
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        print("[debug] dtheta,dy", dtheta, dy)
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)          # 主曲率方向
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)    # 法线方向
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy  # 手抓底部中心点
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.gripper.init_bite * (  # 手抓向物体靠近init_bite米
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        if params['debug_vis']:
                            mlab.figure(bgcolor=(1, 1, 1), size=(640, 480))
                            mlab.clf()
                            self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                         minor_pc, scale_factor=0.001)
                            self.show_grasp_norm_oneside(selected_surface, tmp_grasp_normal * normal_dir, tmp_major_pc * normal_dir,
                                                         minor_pc, scale_factor=0.001)
                            self.show_grasp_norm_oneside(tmp_grasp_bottom_center, tmp_grasp_normal * normal_dir, tmp_major_pc * normal_dir,
                                                         minor_pc, scale_factor=0.001)
                            self.show_points(all_points)
                            mlab.show()

                        # hand_points = self.get_hand_points(tmp_grasp_bottom_center, tmp_grasp_normal, tmp_major_pc)

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, all_points,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, all_points,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, all_points,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])

                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        center_dy = dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)]
                        # we check if the gripper has a potential to collide with the table
                        # by check if the gripper is grasp from a down to top direction
                        finger_top_pos = center_dy[0] + center_dy[1] * self.gripper.hand_depth
                        # [- self.gripper.hand_depth * 0.5] means we grasp objects as a angel larger than 30 degree
                        if finger_top_pos[2] < center_dy[0][2] - self.gripper.hand_depth * 0.5:
                            potential_grasp.append(center_dy)

                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist / params['approach_step'])

                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, point_cloud, hand_points)

                        if is_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step'] * 3
                            # minus 3 means we want the grasp go back a little bitte more.

                            # here we check if the gripper collide with the table.
                            hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                                tmp_grasp_normal,
                                                                tmp_major_pc)[1:]
                            min_finger_end = hand_points_[:, 2].min()
                            min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]

                            safety_dis_above_table = 0.01
                            if min_finger_end < safety_dis_above_table:
                                min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                                x = -min_finger_pos[2]*tmp_grasp_normal[0]/tmp_grasp_normal[2]+min_finger_pos[0]
                                y = -min_finger_pos[2]*tmp_grasp_normal[1]/tmp_grasp_normal[2]+min_finger_pos[1]
                                p_table = np.array([x, y, 0])  # the point that on the table
                                dis_go_back = np.linalg.norm([min_finger_pos, p_table]) + safety_dis_above_table
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back
                            else:
                                # if the grasp is not collide with the table, do not change the grasp
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                            # final check
                            _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, all_points, hand_points)
                            if (len(open_points) > 10) and not is_collide:
                                # here 10 set the minimal points in a grasp, we can set a parameter later
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc,
                                                                  tmp_grasp_bottom_center_modify])
                                if params['debug_vis']:
                                    self.show_points(selected_surface, color='r', scale_factor=.005)
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, all_points, hand_points,
                                                                "p_open", vis=True)
                                break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            print("current amount of sampled surface:", sampled_surface_amount)
            if params['debug_vis']:  # not sampled_surface_amount % 5:
                print("\033[0;32m%s\033[0m" % "Show sampled surface.")
                if len(all_points) > 10000:
                    pc = pcl.PointCloud(all_points)
                    voxel = pc.make_voxel_grid_filter()
                    voxel.set_leaf_size(0.01, 0.01, 0.01)
                    point_cloud = voxel.filter()
                    all_points = point_cloud.to_array()
                self.show_all_grasps(all_points, processed_potential_grasp)
                self.show_points(all_points, scale_factor=0.008)
                mlab.show()
            print("The grasps number got by modified GPG:", len(processed_potential_grasp))
            if len(processed_potential_grasp) >= num_grasps or sampled_surface_amount >= max_num_samples:
                if show_final_grasp:
                    self.show_all_grasps(all_points, processed_potential_grasp)
                    self.show_points(all_points, scale_factor=0.002)
                    mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))
                    table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
                    triangles = [(1, 2, 3), (0, 1, 3)]
                    mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
                                         triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
                    mlab.show()
                return processed_potential_grasp
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp