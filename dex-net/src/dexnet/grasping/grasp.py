#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: MrRen-sdhm

from abc import ABCMeta, abstractmethod
import logging
import coloredlogs
import time
import numpy as np
from numpy.linalg import norm

from autolab_core import RigidTransform

from dexnet import abstractstatic
from dexnet.grasping.contacts import Contact3D

try:
    import mayavi.mlab as mlab
except:
    try:
        import mayavi.mlab as mlab
    except ImportError:
        mlab = []
        logging.error('Failed to import mayavi')

coloredlogs.install(level='INFO')
logger = logging.getLogger(__name__)


class Grasp(object):
    """ Abstract grasp class.

    Attributes
    ----------
    configuration : :obj:`numpy.ndarray`
        vector specifying the parameters of the grasp (e.g. hand pose, opening width, joint angles, etc)
    frame : :obj:`str`
        string name of grasp reference frame (defaults to obj)
    """
    __metaclass__ = ABCMeta
    samples_per_grid = 2  # global resolution for line of action

    @abstractmethod
    def close_fingers(self, obj):
        """ Finds the contact points by closing on the given object.
        
        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to find contacts on
        """
        pass

    @abstractmethod
    def configuration(self):
        """ Returns the numpy array representing the hand configuration """
        pass

    @abstractmethod
    def frame(self):
        """ Returns the string name of the grasp reference frame  """
        pass

    @abstractstatic
    def params_from_configuration(configuration):
        """ Convert configuration vector to a set of params for the class """
        pass

    @abstractstatic
    def configuration_from_params(*params):
        """ Convert param list to a configuration vector for the class """
        pass

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

    def show_surface_points(self, obj):
        self.show_points(obj.cloud)


class PointGrasp(Grasp):
    """ Abstract grasp class for grasps with a point contact model.

    Attributes
    ----------
    configuration : :obj:`numpy.ndarray`
        vector specifying the parameters of the grasp (e.g. hand pose, opening width, joint angles, etc)
    frame : :obj:`str`
        string name of grasp reference frame (defaults to obj)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_line_of_action(start_point, axis, width, num_samples):
        """ Creates a line of action, or the points in space that the grasp traces out, from a point g in world coordinates on an object.

        Returns
        -------
        bool
            whether or not successful
        :obj:`list` of :obj:`numpy.ndarray`
            points in 3D space along the line of action
        """
        pass

    # NOTE: implementation of close_fingers must return success, array of contacts (one per column)


class ParallelJawPtGrasp3D(PointGrasp):
    """ Parallel Jaw point grasps in 3D space.
    """

    def __init__(self, configuration, type='axis', frame='object', grasp_id=None):
        """
        :param configuration:
        :param type: 'axis' use old formate: center+aixs ; 'frame' use new formate: center+frame
               frame: normal+major_pc+minor_pc Note: add by MrRen-sdhm
        :param frame:
        :param grasp_id:
        """
        # get parameters from configuration array
        grasp_center, grasp_axis, grasp_width, grasp_angle, depth, min_grasp_width, normal, minor_pc = \
                                             ParallelJawPtGrasp3D.params_from_configuration(configuration, type)

        self.center_ = grasp_center
        self.axis_ = grasp_axis / np.linalg.norm(grasp_axis)
        self.max_grasp_width_ = grasp_width
        self.depth_ = depth
        self.min_grasp_width_ = min_grasp_width
        self.approach_angle_ = grasp_angle
        self.frame_ = frame
        self.grasp_id_ = grasp_id
        self.normal_ = normal
        self.minor_pc_ = minor_pc

    @property
    def center(self):
        """ :obj:`numpy.ndarray` : 3-vector specifying the center of the jaws """
        return self.center_

    @center.setter
    def center(self, x):
        self.center_ = x

    @property
    def axis(self):
        """ :obj:`numpy.ndarray` : normalized 3-vector specifying the line between the jaws """
        return self.axis_

    @property
    def open_width(self):
        """ float : maximum opening width of the jaws """
        return self.max_grasp_width_

    @property
    def close_width(self):
        """ float : minimum opening width of the jaws """
        return self.min_grasp_width_

    @property
    def depth(self):
        """ float : depth of the close area """
        return self.depth_

    @property
    def approach_angle(self):
        """ float : approach angle of the grasp """
        return self.approach_angle_

    @property
    def normal(self):
        """ float : minor principle curvature of the grasp frame """
        if len(self.normal_) > 0:
            return self.normal_
        else:
            return self.rotated_full_axis[:, 0]

    @normal.setter
    def normal(self, value):
        self._normal_ = value

    @property
    def major_pc(self):
        """ float : minor principle curvature of the grasp frame """
        if len(self.axis_) > 0:
            return self.axis_
        else:
            return self.rotated_full_axis[:, 1]

    @major_pc.setter
    def major_pc(self, value):
        self.axis_ = value

    @property
    def minor_pc(self):
        """ float : minor principle curvature of the grasp frame """
        if len(self.minor_pc_) > 0:
            return self.minor_pc_
        else:
            return self.rotated_full_axis[:, 2]

    @minor_pc.setter
    def minor_pc(self, value):
        self._minor_pc_ = value

    @property
    def configuration(self):
        """ :obj:`numpy.ndarray` : vector specifying the parameters of the grasp as follows
        (grasp_center, grasp_axis, grasp_angle, grasp_width, depth, normal, minor_pc) """
        return ParallelJawPtGrasp3D.configuration_from_params(self.center_, self.axis_, self.max_grasp_width_,
                                                              self.approach_angle_, self.depth_,
                                                              self.min_grasp_width_, self.normal_, self.minor_pc_)

    @property
    def frame(self):
        """ :obj:`str` : name of grasp reference frame """
        return self.frame_

    @property
    def id(self):
        """ int : id of grasp """
        return self.grasp_id_

    @frame.setter
    def frame(self, f):
        self.frame_ = f

    @approach_angle.setter
    def approach_angle(self, angle):
        """ Set the grasp approach angle """
        self.approach_angle_ = angle

    @property
    def endpoints(self):
        """
        Returns
        -------
        :obj:`numpy.ndarray`
            location of jaws(fingers) in 3D space at max opening width
            Note: return the two fingers' tip location
        """
        return self.center_ - (self.max_grasp_width_ / 2.0) * self.axis_, self.center_ + (
                self.max_grasp_width_ / 2.0) * self.axis_,

    @staticmethod
    def distance(g1, g2, alpha=0.05):
        """ Evaluates the distance between two grasps.

        Parameters
        ----------
        g1 : :obj:`ParallelJawPtGrasp3D`
            the first grasp to use
        g2 : :obj:`ParallelJawPtGrasp3D`
            the second grasp to use
        alpha : float
            parameter weighting rotational versus spatial distance

        Returns
        -------
        float
            distance between grasps g1 and g2
        """
        center_dist = np.linalg.norm(g1.center - g2.center)
        axis_dist = (2.0 / np.pi) * np.arccos(np.abs(g1.axis.dot(g2.axis)))
        return center_dist + alpha * axis_dist

    def configuration_from_params(center, axis, width, angle=0, depth=0, min_width=0, normal=[], minor_pc=[]):
        """ Converts grasp parameters to a configuration vector. """
        if np.abs(np.linalg.norm(axis) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        configuration = np.zeros(16)
        configuration[0:3] = center
        configuration[3:6] = axis
        configuration[6] = width
        configuration[7] = angle
        configuration[8] = depth
        configuration[9] = min_width
        if len(normal) == 3:
            configuration[10:13] = normal
        if len(minor_pc) == 3:
            configuration[13:16] = minor_pc
        return configuration

    def params_from_configuration(configuration, type='axis'):
        """ Converts configuration vector into grasp parameters.
        
        Returns
        -------
        grasp_center : :obj:`numpy.ndarray`
            center of grasp in 3D space
        grasp_axis : :obj:`numpy.ndarray`
            normalized axis of grasp in 3D space, major principle curvature
        max_width : float
            maximum opening width of jaws
            Note: should >= hand_outer_diameter-2finger_width, could be hand_outer_diameter-finger_width
        angle : float
            approach angle
        depth : float
            depth of the close area
        min_width : float
            minimum closing width of jaws
        normal : float Note: add by MrRen-sdhm
            approach direction of the grasp frame
        minor_pc : float Note: add by MrRen-sdhm
            minor principle curvature of the grasp frame
        """
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 16):
            raise ValueError('Configuration must be numpy ndarray of size 9 or 16')
        if configuration.shape[0] >= 9:
            min_grasp_width = 0
        else:
            min_grasp_width = configuration[9]
        if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')

        if type=='axis':
            return configuration[0:3], configuration[3:6], configuration[6], configuration[7], configuration[
                8], min_grasp_width, [], []
        elif type=='frame':
            return configuration[0:3], configuration[3:6], configuration[6], configuration[7], configuration[
                8], configuration[9], configuration[10:13], configuration[13:16]

    @staticmethod
    def center_from_endpoints(g1, g2):
        """ Grasp center from endpoints as np 3-arrays """
        grasp_center = (g1 + g2) / 2
        return grasp_center

    @staticmethod
    def axis_from_endpoints(g1, g2):
        """ Normalized axis of grasp from endpoints as np 3-arrays """
        grasp_axis = g2 - g1
        if np.linalg.norm(grasp_axis) == 0:
            return grasp_axis
        return grasp_axis / np.linalg.norm(grasp_axis)

    @staticmethod
    def width_from_endpoints(g1, g2):
        """ Width of grasp from endpoints """
        grasp_axis = g2 - g1
        return np.linalg.norm(grasp_axis)

    @staticmethod
    def grasp_from_endpoints(g1, g2, width=None, approach_angle=0, close_width=0):
        """ Create a grasp from given endpoints in 3D space, making the axis the line between the points.

        Parameters
        ---------
        g1 : :obj:`numpy.ndarray`
            location of the first jaw
        g2 : :obj:`numpy.ndarray`
            location of the second jaw
        width : float
            maximum opening width of jaws
        approach_angle : float
            approach angle of grasp
        close_width : float
            width of gripper when fully closed
        """
        x = ParallelJawPtGrasp3D.center_from_endpoints(g1, g2)
        v = ParallelJawPtGrasp3D.axis_from_endpoints(g1, g2)
        if width is None:
            width = ParallelJawPtGrasp3D.width_from_endpoints(g1, g2)
        return ParallelJawPtGrasp3D(
            ParallelJawPtGrasp3D.configuration_from_params(x, v, width, min_width=close_width, angle=approach_angle))

    @property
    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the 0-degree approach direction, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        if len(self.normal_) > 0 and len(self.minor_pc_) > 0:  # Note: add by MrRen-sdhm
            grasp_axis_x = self.normal_
            grasp_axis_z = self.minor_pc_
        else:
            grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
            if np.linalg.norm(grasp_axis_x) == 0:
                grasp_axis_x = np.array([1, 0, 0])
            grasp_axis_x = grasp_axis_x / norm(grasp_axis_x)
            grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        return R

    @property
    def rotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. X axis points out of the
        gripper palm along the grasp approach angle, Y axis points between the jaws, and the Z axs is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        R = ParallelJawPtGrasp3D._get_rotation_matrix_y(self.approach_angle)
        R = self.unrotated_full_axis.dot(R)
        return R

    @property
    def T_grasp_obj(self):
        """ Rigid transformation from grasp frame to object frame.
        Rotation matrix is X-axis along approach direction, Y axis pointing between the jaws, and Z-axis orthogonal.
        Translation vector is the grasp center.

        Returns
        -------
        :obj:`RigidTransform`
            transformation from grasp to object coordinates
        """
        T_grasp_obj = RigidTransform(self.rotated_full_axis, self.center, from_frame='grasp', to_frame='obj')
        return T_grasp_obj

    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R

    def gripper_pose(self, gripper=None):
        """ Returns the RigidTransformation from the gripper frame to the object frame when the gripper is executing the
        given grasp.
        Differs from the grasp reference frame because different robots use different conventions for the gripper
        reference frame.
        
        Parameters
        ----------
        gripper : :obj:`RobotGripper`
            gripper to get the pose for

        Returns
        -------
        :obj:`RigidTransform`
            transformation from gripper frame to object frame
        """
        if gripper is None:
            T_gripper_grasp = RigidTransform(from_frame='gripper', to_frame='grasp')
        else:
            T_gripper_grasp = gripper.T_grasp_gripper

        T_gripper_obj = self.T_grasp_obj * T_gripper_grasp
        return T_gripper_obj

    def create_line_of_action(start_point, axis, width, num_samples):
        """
        Creates a straight line of action, from a given point and direction in world coords

        Parameters
        ----------
        start_point : 3x1 :obj:`numpy.ndarray`
            start point to create the line of action
        axis : normalized 3x1 :obj:`numpy.ndarray`
            normalized numpy 3 array of grasp direction
        width : float
            the grasp width
        num_samples : int
            number of discrete points along the line of action

        Returns
        -------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarrays`
            coordinates to pass through in 3D space for contact checking
        """
        num_samples = max(num_samples, 3)  # always at least 3 samples
        line_of_action = [start_point + t * axis for t in np.linspace(0, float(width), num=num_samples)]
        return line_of_action

    def close_fingers(self, obj, check_approach=False, approach_dist=1.0, vis=False):
        """ Steps along grasp axis to find the locations of contact with an object

        Parameters
        ----------
        obj : :obj:`GraspableObjectPcd`
            object to close fingers on
        vis : bool
            whether or not to plot the line of action and contact points
        check_approach : bool
            whether or not to check if the contact points can be reached
        approach_dist : float
            how far back to check the approach distance, only if checking the approach is set
        
        Returns
        -------
        success : bool
            whether or not contacts were found
        c1 : :obj:`Contact3D`
            the contact point for jaw 1
        c2 : :obj:`Contact3D`
            the contact point for jaw 2
        """

        start = time.perf_counter()
        r_ball = 0.003
        num_samples = int(self.max_grasp_width_ / r_ball)
        num_samples_depth = int(num_samples/2)

        # get grasp endpoints
        g1_world, g2_world = self.endpoints
        g1_world_list = [g1_world + t * (-self.normal) for t in np.linspace(0, self.depth, num=num_samples_depth)]
        g2_world_list = [g2_world + t * (-self.normal) for t in np.linspace(0, self.depth, num=num_samples_depth)]

        # show endpoints
        if False:
            self.show_surface_points(obj)
            self.show_arrow(self.center, self.normal, 'r', 0.02)
            self.show_points(np.array([g1_world_list[0], g2_world_list[0]]), color='r', scale_factor=0.003)
            self.show_points(np.array([g1_world_list[-1], g2_world_list[-1]]), color='b', scale_factor=0.003)
            self.show_points(np.array(g1_world_list), color='g', scale_factor=0.001)
            self.show_points(np.array(g2_world_list), color='g', scale_factor=0.001)
            mlab.show()

        start_contact_flag = False
        contacts_list = []
        for i in range(num_samples_depth):
            g1_world, g2_world = (g1_world_list[i], g2_world_list[i])

            # get line of action
            line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.open_width, num_samples)
            line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.open_width, num_samples)

            # find contacts
            c1_found, c1 = self.find_contact(line_of_action1, obj, r_ball)
            c2_found, c2 = self.find_contact(line_of_action2, obj, r_ball)

            contacts_found = c1_found and c2_found

            if contacts_found:
                start_contact_flag = True  # find the first pair of contacts
                contacts_list.append([c1, c2])

                if vis:
                    self.show_points(line_of_action1, 'r', 0.0015)
                    self.show_points(line_of_action2, 'b', 0.0015)
                    self.show_points(np.array(g1_world_list), color='y', scale_factor=0.0015)
                    self.show_points(np.array(g2_world_list), color='y', scale_factor=0.0015)
                    self.show_surface_points(obj)
                    self.show_points(np.array([c1.point, c2.point]), color='p', scale_factor=0.003)
                    self.show_arrow(c1.point, c1.in_direction, 'r', scale_factor=0.02)
                    self.show_arrow(c2.point, c2.in_direction, 'g', scale_factor=0.02)
                    # mlab.show()

            # if (start_contact_flag and not contacts_found) or (start_contact_flag and i == num_samples_depth):  # find the last contacts
            #     if len(contacts_list) < 2:  # too few contacts
            #         logger.info("too few contacts")
            #         return False, [None, None]
            #
            #     # find the pair of contacts which have max distance
            #     dist_list = []
            #     for c1, c2 in contacts_list:
            #         dist_list.append(np.linalg.norm(c1.point-c2.point))  # cal the distance
            #     max_index = dist_list.index(max(dist_list))  # cal the max distance's index
            #
            #     if vis:
            #         self.show_points(np.array([g1_world, g2_world]), color='g', scale_factor=0.005)
            #         self.show_points(np.array([contacts_list[max_index][0].point, contacts_list[max_index][1].point]),
            #                          color='b', scale_factor=0.005)
            #         normal1 = contacts_list[max_index][0].normal
            #         normal2 = contacts_list[max_index][1].normal
            #         self.show_arrow(contacts_list[max_index][0].point, normal1, 'y', scale_factor=0.02)
            #         self.show_arrow(contacts_list[max_index][1].point, normal2, 'y', scale_factor=0.02)
            #         mlab.show()
            #
            #     # logger.debug("find first contact took %.4f" % (start_contact - start))
            #     logger.debug("close finger took %.4f" % (time.perf_counter()-start))
            #     return True, contacts_list[max_index]

        # print("[DEBUG] len(contacts_list)", len(contacts_list))
        if len(contacts_list) >= 2:  # at least found 2 contacts
            # find the pair of contacts which have max distance
            dist_list = []
            for c1, c2 in contacts_list:
                dist_list.append(np.linalg.norm(c1.point - c2.point))  # cal the distance
            max_index = dist_list.index(max(dist_list))  # cal the max distance's index

            if vis:
                self.show_points(np.array([g1_world, g2_world]), color='g', scale_factor=0.005)
                self.show_points(np.array([contacts_list[max_index][0].point, contacts_list[max_index][1].point]),
                                 color='b', scale_factor=0.005)
                normal1 = contacts_list[max_index][0].normal
                normal2 = contacts_list[max_index][1].normal
                self.show_arrow(contacts_list[max_index][0].point, normal1, 'y', scale_factor=0.02)
                self.show_arrow(contacts_list[max_index][1].point, normal2, 'y', scale_factor=0.02)
                mlab.show()

            if max(dist_list) > self.min_grasp_width_:
                logger.debug("close finger took %.4f" % (time.perf_counter() - start))
                return True, contacts_list[max_index]
            else:
                print("max(dist_list): ", max(dist_list))
                print("close_width: ", self.min_grasp_width_)
                logger.info("distance too short")

        logger.info("not found contacts")
        logger.debug("close finger took %.4f" % (time.perf_counter() - start))
        return False, [None, None]

    def find_contact(self, line_of_action, graspable, r_ball=0.001):
        """
        Find the point at which a point traveling along a given line of action hits a surface.

        Parameters
        ----------
        line_of_action : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            the points visited as the fingers close (grid coords)
        graspable : :obj:`GraspableObjectPcd`
            to check contacts on
        r_ball : radius to search the neibor point

        Returns
        -------
        contact_found : bool
            whether or not the point contacts the object surface
        contact : :obj:`Contact3D`
            found along line of action (None if contact not found)
        """
        contact_found = False
        contact = None
        point_world = None
        point_indice = None

        # step along line of action, get points on surface when possible
        t = time.perf_counter()
        for i in range(len(line_of_action)):
            point_world = line_of_action[i].astype(np.float32)
            belong_to, point_indice = graspable.belong_to(point_world, r_ball)
            if belong_to:
                contact_found = True
                break

        if contact_found:
            in_direction = line_of_action[-1] - line_of_action[0]
            in_direction = in_direction / np.linalg.norm(in_direction)

            """ creat Contact3D then correct the normal of contact point """
            contact = Contact3D(graspable, point_world, point_indice, in_direction=in_direction)

            if False:
                self.show_points(line_of_action[0], 'y', 0.005)  # start point
                self.show_arrow(point_world, in_direction, 'g')
                self.show_arrow(point_world, graspable.normals[point_indice], 'r')  # raw normal
                self.show_arrow(point_world, contact.normal, 'b')  # show corrected normal
                self.show_surface_points(graspable)
                self.show_points(line_of_action, 'r', 0.001)
                self.show_points(point_world, color='r', scale_factor=0.005)
                mlab.show()

            # print("[DEBUG] contact_found")

        logger.debug("find contact take: %f" % (time.perf_counter()-t))
        return contact_found, contact
