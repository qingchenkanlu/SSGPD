# -*- coding: utf-8 -*-

import json
import os

GRIPPER_PARAMS_FILENAME = 'params.json'


class RobotGripper(object):
    """ Robot gripper wrapper for collision checking and encapsulation of grasp parameters (e.g. width, finger radius, etc)
    Note: The gripper frame should be the frame used to command the physical robot
    
    Attributes
    ----------
    name : :obj:`str`
        name of gripper
    params : :obj:`dict`
        set of parameters for the gripper, at minimum (finger_radius and grasp_width)
    """

    def __init__(self, name, params):
        self.name = name

        for key, value in list(params.items()):
            setattr(self, key, value)

    @staticmethod
    def load(gripper_name, gripper_dir='data/grippers'):
        """ Load the gripper specified by gripper_name.

        Parameters
        ----------
        gripper_name : :obj:`str`
            name of the gripper to load
        gripper_dir : :obj:`str`
            directory where the gripper files are stored

        Returns
        -------
        :obj:`RobotGripper`
            loaded gripper objects
        """
        
        f = open(os.path.join(os.path.join(gripper_dir, gripper_name, GRIPPER_PARAMS_FILENAME)), 'r')
        params = json.load(f)

        return RobotGripper(gripper_name, params)
