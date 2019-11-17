#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : MrRen-sdhm
# File Name  : grasps_save_read.py

import numpy as np
import os
import logging
import pickle

import pcl

from gripper import RobotGripper
from autolab_core import YamlConfig
from Generator.grasp.grasp_sampler import GpgGraspSampler

try:
    import mayavi.mlab as mlab
except:
    try:
        import mayavi.mlab as mlab
    except ImportError:
        mlab = []
        logging.error('Failed to import mayavi')


def show_grasps_info(grasps):
    min_scores = np.linspace(0.0, 3.2, num=30)
    print("total grasps: %d" % len(grasps))
    print("[min-max]: num")
    for min_score in min_scores:
        max_score = min_score + 0.1

        m = np.array(grasps)
        m = m[m[:, 1] <= max_score]
        m = m[m[:, 1] >= min_score]
        print("[%.1f-%.1f]: %d" % (min_score, max_score, len(m)))


def show_grasps(obj, grasps):
    m = np.array(grasps)
    m_good = m[m[:, 1] <= 0.6]  # NOTE: 摩擦系数<=0.6为质量高的抓取
    m_good = m_good[m_good[:, 1] >= 0.0]
    m_good = m_good[np.random.choice(len(m_good), size=5, replace=True)]  # 随机选择25个显示
    m_bad = m[m[:, 1] >= 0.7]  # NOTE: 摩擦系数>=0.7为质量低的抓取
    m_bad = m_bad[m_bad[:, 1] <= 3.0]
    m_bad = m_bad[np.random.choice(len(m_bad), size=5, replace=True)]  # 随机选择25个显示

    ags = GpgGraspSampler(gripper, yaml_config)

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1024, 768))
    for good in m_good:
        ags.display_grasps3d(good[0], 'g')
    ags.show_points(obj)
    mlab.title("good", size=0.5)
    # mlab.show()

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1024, 768))
    for bad in m_bad:
        ags.display_grasps3d(bad[0], 'r')
    ags.show_points(obj)
    mlab.title("bad", size=0.5)
    mlab.show()


def show_grasps_range(obj, grasps, min_score=0.0, max_score=3.2):
    m = np.array(grasps)
    m = m[m[:, 1] <= max_score]  # NOTE: 摩擦系数<=0.6为质量高的抓取
    m = m[m[:, 1] >= min_score]
    if not len(m) > 0:
        return
    m = m[np.random.choice(len(m), size=5, replace=True)]  # 随机选择25个显示

    ags = GpgGraspSampler(gripper, yaml_config)

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1024, 768))
    for grasp in m:
        ags.display_grasps3d(grasp[0], 'g')
    ags.show_points(obj)
    mlab.title("%.1f-%.1f" % (min_score, max_score), size=0.5)
    mlab.show()


def grasps_save(grasps, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(grasps, f)


def grasps_read(filename):
    grasps = pickle.load(open(filename, 'rb'))
    if isinstance(grasps, list):
        return grasps
    else:
        return [grasps]


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    yaml_config = YamlConfig(home_dir + "/Projects/GPD_PointNet/dex-net/test/sample_config.yaml")
    gripper = RobotGripper.load('robotiq_85', home_dir + "/Projects/GPD_PointNet/dex-net/data/grippers")
    pickle_path = home_dir + "/grasps/grasps_with_score.pickle"
    obj_path = home_dir + "/Projects/GPD_PointNet/normal_estimator/cloud.pcd"

    obj = pcl.load(obj_path).to_array()
    grasps = grasps_read(pickle_path)

    show_grasps(obj, grasps)
    # show_grasps_range(obj, grasps, min_score=0.0, max_score=0.5)
    show_grasps_info(grasps)

    min_scores = np.linspace(0.0, 3.2, num=30)
    # print(min_scores)
    for min_score in min_scores:
        show_grasps_range(obj, grasps, min_score=min_score, max_score=min_score+0.1)


