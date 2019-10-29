#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : sdhm
# Description:
# Date       : 06/08/2019 14:22 PM
# File Name  : cal_grasp_from_file.py
# Note: this file is written in Python2

import numpy as np
import voxelgrid
import pcl
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSamplerPclPcd
from dexnet.grasping import AntipodalGraspSampler
import os
from pyquaternion import Quaternion
import sys
from os import path
import time
import multiprocessing as mp


try:
    from mayavi import mlab
except ImportError:
    print("Can not import mayavi")
    mlab = None
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath("__file__")))))
sys.path.append(os.environ['HOME'] + "/code/grasp-pointnet/PointNetGPD")


# global config:
yaml_config = YamlConfig(os.environ['HOME'] + "/Projects/GPD_PointNet/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
gripper = RobotGripper.load(gripper_name, os.environ['HOME'] + "/Projects/GPD_PointNet/dex-net/data/grippers")
ags = GpgGraspSamplerPclPcd(gripper, yaml_config)
# ags = AntipodalGraspSampler(gripper, yaml_config)
value_fc = 0.4  # no use, set a random number
num_grasps = 40
num_workers = 20
max_num_samples = 150
n_voxel = 500

minimal_points_send_to_point_net = 20
marker_life_time = 8

show_bad_grasp = False
save_grasp_related_file = False

using_mp = False
show_final_grasp = True


def remove_table_points(points_voxel_, vis=False):
    """
    移除平面
    :param points_voxel_: 体素栅格下采样之后的点云
    :param vis:
    :return: 移除平面之后的点云
    """
    xy_unique = np.unique(points_voxel_[:, 0:2], axis=0)
    new_points_voxel_ = points_voxel_
    pre_del = np.zeros([1])
    for i in range(len(xy_unique)):
        tmp = []
        for j in range(len(points_voxel_)):
            if np.array_equal(points_voxel_[j, 0:2], xy_unique[i]):
                tmp.append(j)
        print(len(tmp))
        if len(tmp) < 3:
            tmp = np.array(tmp)
            pre_del = np.hstack([pre_del, tmp])
    if len(pre_del) != 1:
        pre_del = pre_del[1:]
        new_points_voxel_ = np.delete(points_voxel_, pre_del, 0)
    print("Success delete [[ {} ]] points from the table!".format(len(points_voxel_) - len(new_points_voxel_)))

    if vis:
        p = points_voxel_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = new_points_voxel_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))  # plot 0 point
        mlab.show()
    return new_points_voxel_


def get_voxel_fun(points_, n):
    """
    体素栅格下采样
    :param points_: 点云中的点序列
    :param n: 体素大小
    :return: 降采样之后的点
    """
    get_voxel = voxelgrid.VoxelGrid(points_, n_x=n, n_y=n, n_z=n)
    get_voxel.compute()
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


def cal_grasp(cloud_, cam_pos_):
    """
    抓取姿态生成
    :param cloud_: 点云
    :param cam_pos_: 摄像头姿态
    :return:
    """
    points_ = cloud_.to_array()
    # for i in range(points_.shape[0]):
    #     if points_[i][0] < 2:
    #         print(points_[i])

    print("points_ shape", points_.shape, points_.dtype)
    # mlab.points3d(points_[:, 0], points_[:, 1], points_[:, 2], scale_factor=0.001, color=(0, 1, 1))  # 显示点云
    # mlab.show()
    # exit(1)

    # begin voxel points
    n = n_voxel  # parameter related to voxel method
    # gpg improvements, highlights: flexible n parameter for voxelizing.
    points_[:, 0] = points_[:, 0] + 0.025  # liang: as the kinect2 is not well calibrated, here is a work around
    points_[:, 2] = points_[:, 2]  # + 0.018  # liang: as the kinect2 is not well calibrated, here is a work around

    points_voxel_ = points_
    # points_voxel_ = get_voxel_fun(points_, n)  # point cloud down sample
    # # 点数过少, 调整降采样参数
    # if len(points_) < 2000:  # should be a parameter
    #     while len(points_voxel_) < len(points_)-15:
    #         points_voxel_ = get_voxel_fun(points_, n)
    #         n = n + 100
    #         print("the voxel has {} points, we want get {} points".format(len(points_voxel_), len(points_)))

    mlab.points3d(points_voxel_[:, 0], points_voxel_[:, 1], points_voxel_[:, 2], scale_factor=0.001, color=(0, 1, 1))
    mlab.show()

    print("the voxel has {} points, we want get {} points".format(len(points_voxel_), len(points_)))

    points_ = points_voxel_  # 降采样之后的点云
    remove_points = False
    if remove_points:
        points_ = remove_table_points(points_, vis=True)

    point_cloud = pcl.PointCloud(points_)  # 传入pcl处理
    # 计算表面法线
    print("Calculate point cloud normal...")
    norm = point_cloud.make_NormalEstimation()
    norm.set_KSearch(30)  # critical parameter when calculating the norms
    normals = norm.compute()
    surface_normal = normals.to_array()
    surface_normal = surface_normal[:, 0:3]
    # 处理法线方向
    vector_p2cam = cam_pos_ - points_
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)
    tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    tmp = np.ones([len(angel), 3])
    tmp[wrong_dir_norm, :] = -1
    surface_normal = surface_normal * tmp
    select_point_above_table = 0.010
    #  modify of gpg: make it as a parameter. avoid select points near the table.
    points_for_sample = points_[np.where(points_[:, 2] > select_point_above_table)[0]]
    if len(points_for_sample) == 0:
        print("Can not select point, maybe the point cloud is too low?")
        return [], points_, surface_normal
    yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
    if not using_mp:
        print("Begin cal grasps using single thread, slow!")
        grasps_together_ = ags.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps,
                                             max_num_samples=max_num_samples, show_final_grasp=show_final_grasp)
    else:
        # begin parallel grasp:
        print("Begin cal grasps using parallel!")

        def grasp_task(num_grasps_, ags_, queue_):
            ret = ags_.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_,
                                     max_num_samples=max_num_samples, show_final_grasp=show_final_grasp)
            queue_.put(ret)

        queue = mp.Queue()
        num_grasps_p_worker = int(num_grasps/num_workers)
        workers = [mp.Process(target=grasp_task, args=(num_grasps_p_worker, ags, queue)) for _ in range(num_workers)]
        [i.start() for i in workers]

        grasps_together_ = []
        for i in range(num_workers):
            grasps_together_ = grasps_together_ + queue.get()
        print("Finish mp processing!")
    print("Grasp sampler finish, generated {} grasps.".format(len(grasps_together_)))
    return grasps_together_, points_, surface_normal


def check_collision_square(grasp_bottom_center, approach_normal, binormal,
                           minor_pc, points_, p, way="p_open"):
    """
    碰撞检测
    :param grasp_bottom_center:
    :param approach_normal:
    :param binormal:
    :param minor_pc:
    :param points_:
    :param p:
    :param way:
    :return:
    """
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    matrix_ = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
    grasp_matrix = matrix_.T
    points_ = points_ - grasp_bottom_center.reshape(1, 3)
    tmp = np.dot(grasp_matrix, points_.T)
    points_g = tmp.T
    use_dataset_py = True
    if not use_dataset_py:
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
    # for the way of pointGPD/dataset.py:
    else:
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth
        z_limit = width / 4
        y_limit = width / 2
        x1 = points_g[:, 0] > 0
        x2 = points_g[:, 0] < x_limit
        y1 = points_g[:, 1] > -y_limit
        y2 = points_g[:, 1] < y_limit
        z1 = points_g[:, 2] > -z_limit
        z2 = points_g[:, 2] < z_limit
        a = np.vstack([x1, x2, y1, y2, z1, z2])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

    vis = False
    if vis:
        p = points_g
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        p = points_g[points_in_area]
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.005, color=(0, 1, 0))
        mlab.show()

    return has_p, points_in_area, points_g


def collect_pc(grasp_, pc):
    """
    获取手抓坐标系下的点云
    grasp_bottom_center, normal, major_pc, minor_pc
    :param grasp_:
    :param pc:
    :return:
    """
    grasp_num = len(grasp_)
    grasp_ = np.array(grasp_)
    grasp_ = grasp_.reshape(-1, 5, 3)  # prevent to have grasp that only have number 1
    grasp_bottom_center = grasp_[:, 0]
    approach_normal = grasp_[:, 1]
    binormal = grasp_[:, 2]
    minor_pc = grasp_[:, 3]

    in_ind_ = []
    in_ind_points_ = []
    p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    for i_ in range(grasp_num):
        has_p, in_ind_tmp, points_g = check_collision_square(grasp_bottom_center[i_], approach_normal[i_],
                                                             binormal[i_], minor_pc[i_], pc, p)
        in_ind_.append(in_ind_tmp)
        in_ind_points_.append(points_g[in_ind_[i_]])
    return in_ind_, in_ind_points_


def check_hand_points_fun(real_grasp_):
    ind_points_num = []
    for i in range(len(real_grasp_)):
        grasp_bottom_center = real_grasp_[i][4]
        approach_normal = real_grasp_[i][1]
        binormal = real_grasp_[i][2]
        minor_pc = real_grasp_[i][3]
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    binormal, minor_pc, points,
                                                                    local_hand_points, "p_open")
        ind_points_num.append(len(ind_points_tmp))
    print(ind_points_num)
    file_name = "./generated_grasps/real_points/" + str(np.random.randint(300)) + str(len(real_grasp_)) + ".npy"
    np.save(file_name, np.array(ind_points_num))


def remove_grasp_outside_tray(grasps_, points_):
    x_min = points_[:, 0].min()
    x_max = points_[:, 0].max()
    y_min = points_[:, 1].min()
    y_max = points_[:, 1].max()
    valid_grasp_ind_ = []
    for i in range(len(grasps_)):
        grasp_bottom_center = grasps_[i][4]
        approach_normal = grasps_[i][1]
        major_pc = grasps_[i][2]
        hand_points_ = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
        finger_points_ = hand_points_[[1, 2, 3, 4, 9, 10, 13, 14], :]
        # aa = points_[:, :2] - finger_points_[0][:2]  # todo： work of remove outside grasp not finished.

        # from IPython import embed;embed()
        a = finger_points_[:, 0] < x_min
        b = finger_points_[:, 0] > x_max
        c = finger_points_[:, 1] < y_min
        d = finger_points_[:, 1] > y_max
        if np.sum(a) + np.sum(b) + np.sum(c) + np.sum(d) == 0:
            valid_grasp_ind_.append(i)
    grasps_inside_ = [grasps_[i] for i in valid_grasp_ind_]
    print("gpg got {} grasps, after remove grasp outside tray, {} grasps left".format(len(grasps_),
                                                                                              len(grasps_inside_)))
    return grasps_inside_


if __name__ == '__main__':
    """
    definition of gotten grasps:

    grasp_bottom_center = grasp_[0]
    approach_normal = grasp_[1]
    binormal = grasp_[2]
    """

    cam_pos = [0, 0, 0]
    # pcd_file = "/home/sdhm/Projects/GPD_PointNet/PointNetGPD/data/ycb_rgbd/003_cracker_box/clouds/pc_NP1_NP5_60.pcd"
    pcd_file = "/home/sdhm/Projects/kinect2_cloud_samples/data/1/0001_cloud.pcd"

    cloud = pcl.load(pcd_file)

    real_grasp, points, normals_cal = cal_grasp(cloud, cam_pos)  # 获取抓取姿态
