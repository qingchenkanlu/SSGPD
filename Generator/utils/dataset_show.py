#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : MrRen-sdhm
# File Name  : grasps_show.py

import numpy as np
import logging
import glob

from gripper import RobotGripper
from autolab_core import YamlConfig
from Generator.grasp.grasp_sampler import GpgGraspSampler


def load_dataset(dataset_name):
    dataset = None
    if dataset_name == "fusion":
        dataset = load_dataset_fusion()
    elif dataset_name == "ycb":
        dataset = load_dataset_ycb()
    return dataset


def load_dataset_fusion():
    file_dir = "../../Dataset/fusion/"
    obj_name = "2018-04-24-15-57-16"
    obj_path = file_dir + obj_name
    dataset_path = glob.glob(obj_path + "/grasps_with_score*.npy")[0]
    print(dataset_path)
    dataset = np.load(dataset_path, allow_pickle=True)
    print(dataset)
    return dataset


def load_dataset_ycb():
    file_dir = "../../Dataset/ycb/ycb_meshes_google/"
    obj_name = "006_mustard_bottle"
    obj_path = file_dir + obj_name
    dataset_path = glob.glob(obj_path + "/grasps_with_score*.npy")[0]
    print(dataset_path)

    dataset = np.load(dataset_path, allow_pickle=True)
    print(dataset)
    return dataset


if __name__ == '__main__':
    sample_config = YamlConfig("../config/sample_config.yaml")
    gripper = RobotGripper.load("../config/gripper_params.yaml")
    ags = GpgGraspSampler(gripper, sample_config)
    hand_points = ags.get_hand_points([0, 0, 0], [1, 0, 0], [0, 1, 0])  # hand in origion coordinate

    dataset = load_dataset('fusion')
    for data in dataset:
        score = data[1]
        print("score:", score)
        if score < 0.4:
            ags.show_points(data[0])
            ags.show_grasp_3d(hand_points, color='g')
            ags.show_origin(0.03)
            ags.show(str(score))



