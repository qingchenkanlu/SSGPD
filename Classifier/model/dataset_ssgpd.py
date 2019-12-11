#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : MrRen-sdhm
# File Name  : dataset_generator.py
# load dataset for torch.

import os
import sys
import glob

import numpy as np
import open3d as o3d  # should import befor torch: https://github.com/intel-isl/Open3D/pull/1262
import torch
import torch.utils.data
import torch.nn as nn


class OneViewDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, grasp_points_num, dataset_path, tag):
        self.grasp_points_num = grasp_points_num
        self.dataset_path = dataset_path
        self.tag = tag

        fls_grasp = []
        for dirpath, dirnames, files in os.walk(dataset_path):
            for f in files:
                if f.endswith('.pickle'):
                    fls_grasp.append(os.path.join(dirpath, f))
        print("fls_grasp", fls_grasp)

        # load dataset of all obj
        dataset = []
        for fl_grasp in fls_grasp:
            obj_dir = os.path.dirname(fl_grasp)
            print("obj_dir", obj_dir)
            fls_dataset = glob.glob(os.path.join(obj_dir, 'dataset_%s_*.npy' % self.tag))
            print("fl_dataset", fls_dataset)

            dataset_ = []
            for fl_dataset in fls_dataset:
                dataset_.append(np.load(fl_dataset, allow_pickle=True))
            dataset_ = np.concatenate(dataset_)
            dataset.append(dataset_)

        self.dataset = np.concatenate(dataset)
        np.random.shuffle(self.dataset)  # 打乱数据
        self.amount = len(self.dataset)

        # 检查数据分布
        bad_num = good_num = 0
        for data in self.dataset:
            if data[1] == 0:
                bad_num += 1
            elif data[1] == 1:
                good_num += 1
        print("[DEBUG] data num:%d bad_num:%d good_num: %d in dataset" % (self.amount, bad_num, good_num))

    def __getitem__(self, index):
        grasp_pc = self.dataset[index][0]
        label = self.dataset[index][1]

        # 点数不够则有放回采样, 点数太多则随机采样
        if len(grasp_pc) > self.grasp_points_num:
            print("[DEBUG] points in grasp_pc:%d > grasp_points_num:%d | Downsampling." % (len(grasp_pc), self.grasp_points_num))
            grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num, replace=False)].T
        else:
            print("[DEBUG] points in grasp_pc:%d < grasp_points_num:%d | Upsampling." % (len(grasp_pc), self.grasp_points_num))
            grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num, replace=True)].T

        return grasp_pc, label

    def __len__(self):
        return self.amount


if __name__ == '__main__':
    from autolab_core import YamlConfig
    from Generator.grasp.grasp_sampler import GpgGraspSampler
    from Generator.grasp.gripper import RobotGripper

    curr_path = os.path.dirname(os.path.abspath(__file__))
    print("[DEBUG] curr_path", curr_path)
    sample_config = YamlConfig(curr_path + "/../../Generator/config/sample_config.yaml")
    gripper = RobotGripper.load(curr_path + "/../../Generator/config/gripper_params.yaml")
    ags = GpgGraspSampler(gripper, sample_config)
    hand_points = ags.get_hand_points([0, 0, 0], [1, 0, 0], [0, 1, 0])  # hand in origion coordinate

    dataset = OneViewDatasetLoader(
        grasp_points_num=1000,
        dataset_path="../../Dataset/fusion",
        tag='train'
    )

    cnt = 0
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        if data is not None:
            print("[debug] grasp_pc", data[0], data[0].shape)

            # Note: visualize the data
            ags.show_points(data[0].T)
            ags.show_grasp_3d(hand_points, color='g')
            ags.show_origin(0.03)
            ags.show(str(data[1]))
            cnt += 1
    print("[INFO] have {} valid grasp in the dataset({}).".format(cnt, dataset.__len__()))

    # train_loader = torch.utils.data.DataLoader(
    #     OneViewDatasetLoader(
    #         grasp_points_num=1000,
    #         dataset_path="../../Dataset/fusion",
    #         tag='train',
    #     ),
    #     batch_size=64,
    #     num_workers=32,
    #     pin_memory=True,
    #     shuffle=True,
    #     drop_last=True,  # fix bug: ValueError: Expected more than 1 value per channel when training
    # )
    #
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print("data", data, data.shape, "target", target)
    #     pass
