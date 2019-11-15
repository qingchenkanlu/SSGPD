# -*- coding: utf-8 -*-

from autolab_core import YamlConfig


class RobotGripper(object):
    """ Robot gripper wrapper for collision checking and encapsulation of grasp parameters (e.g. width, finger radius, etc)
    """

    def __init__(self, yaml_config):
        for key, value in list(yaml_config.config.items()):
            setattr(self, key, value)
        self.hand_outer_diameter = self.max_width + 2*self.finger_width

    @staticmethod
    def load(dir_):
        """ Load the gripper specified by gripper_name.

        Parameters
        ----------
        dir_ : :obj:`str`
            directory where the gripper files are stored

        Returns
        -------
        :obj:`RobotGripper`
            loaded gripper objects
        """

        yaml_config = YamlConfig(dir_)

        return RobotGripper(yaml_config)
