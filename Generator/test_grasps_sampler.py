#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: MrRen-sdhm

import os
import time
import pickle
import numpy as np
import open3d as o3d

from quality import PointGraspMetrics3D
from grasp_sampler import GpgGraspSampler
from graspable_object import GraspableObject
from gripper import RobotGripper
from autolab_core import YamlConfig
from Generator.utils.grasps_show import grasps_read

import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

DATASET = "ycb"  # ["fusion", "ycb"]


def grasps_save(grasps, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(grasps, f)


def test_grasp_sample():
    reload = False
    grasps = None
    # Test GpgGraspSamplerPcd
    ags = GpgGraspSampler(gripper, sample_config)
    if not reload:
        min_x = 0.002 if DATASET == "fusion" else None  # Note
        grasps = ags.sample_grasps(obj, num_grasps=5000, max_num_samples=10, min_x=min_x, vis=True)
        grasps_save(grasps, "/home/sdhm/grasps/test_pcd")
        # exit()

    """ load grasp from file """
    if reload:
        grasps = grasps_read('/home/sdhm/grasps/all.pickle')

    if False:
        ags.display_grasps3d(grasps)
        ags.show_surface_points(obj, 'r')
        ags.show()

    mark = 0
    start = time.perf_counter()
    fc_list = [4.0, 3.0, 2.0, 1.7, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
    good_count_perfect = np.zeros(len(fc_list), dtype=int)
    contacts_not_found_num = 0
    contacts_found_not_force_closure_num = 0
    proccessed_num = 0
    grasps_save_ls = []
    grasps_with_score = []
    for grasp in grasps:
        proccessed_num += 1
        tmp, is_force_closure = False, False
        contacts_found, contacts = grasp.close_fingers(obj)
        # continue

        if not contacts_found:
            print("[DEBUG] contact_not_found")
            # ags.show_surface_points(obj)
            # ags.display_grasps3d([grasp], 'g')
            # ags.show()
            grasps_save_ls.append(grasp)
            # 未找到接触点, 跳过 FIXME:将这些抓取的摩擦系数设为无穷大
            contacts_not_found_num += 1
            continue
        print("good cnt[%d/%d]" % (proccessed_num, len(grasps)), good_count_perfect)

        for ind_, value_fc in enumerate(fc_list):  # 为每个摩擦系数分配抓取姿态
            value_fc = round(value_fc, 2)
            tmp = is_force_closure
            is_force_closure, _ = PointGraspMetrics3D.grasp_quality(grasp, obj, value_fc, contacts=contacts)

            # print("[INFO] is_force_closure:", bool(is_force_closure), "value_fc:", value_fc, "tmp:", bool(tmp))
            if tmp and not is_force_closure:  # 前一个摩擦系数下为力闭合, 当前摩擦系数下非力闭合, 即找到此抓取对应的最小摩擦系数
                # print("[DEBUG] tmp and not is_force_closure,value_fc:", value_fc, "ind_:", ind_)
                good_count_perfect[ind_-1] += 1  # 前一个摩擦系数最小
                grasps_with_score.append((grasp, round(fc_list[ind_-1], 2)))  # 保存前一个抓取

                # if np.isclose(value_fc, 0.2):
                #     # ags.show_surface_points(obj)
                #     ags.display_grasps3d([grasp], 'g')

                #     ags.show()

                break  # 找到即退出
            elif is_force_closure and np.isclose(value_fc, fc_list[-1]):  # 力闭合并且摩擦系数最小
                # print("[DEBUG] is_force_closure and value_fc == fc_list[-1]")
                good_count_perfect[ind_] += 1  # 已无更小摩擦系数, 此系数最小
                grasps_with_score.append((grasp, value_fc))

                # ags.display_grasps3d([grasp], 'b')

                break  # 找到即退出

            if not is_force_closure and np.isclose(value_fc, fc_list[-1]):  # 判断结束还未找到对应摩擦系数,并且找到一对接触点
                print("[DEBUG] not is_force_closure but contacts_found")
                contacts_found_not_force_closure_num += 1
                grasps_with_score.append((grasp, 3.2, 0))

                # grasps_save(grasp, "/home/sdhm/grasps/%s" % str(mark))
                # print("[DEBUG] save grasp to %s.pickle" % str(mark))
                # mark += 1

                # ags.display_grasps3d([grasp], 'g')
                # ags.show_surface_points(obj, color='r')
                # ags.show()

                break  # 找到即退出

    print("\n\ngood cnt", good_count_perfect)
    print("proccessed grasp num:", len(grasps))
    print("good_count_perfect num:", int(good_count_perfect.sum()))
    print("contacts_not_found num:", contacts_not_found_num)
    print("contacts_found_not_force_closure num:", contacts_found_not_force_closure_num)
    print("classify took {:.2f} s".format(time.perf_counter()-start))

    grasps_save(grasps_with_score, "/home/sdhm/grasps/grasps_with_score")
    grasps_save(grasps_save_ls, "/home/sdhm/grasps/contact_not_found")
    grasps_save(grasps, "/home/sdhm/grasps/all")

    # ags.show_surface_points(obj, color='r')
    # ags.show()
    return True


if __name__ == '__main__':
    file_dir = None  # Note
    if DATASET == "fusion":
        file_dir = "./utils/extract_normals"
    elif DATASET == "ycb":
        file_dir = "./test/data"

    sample_config = YamlConfig("./config/sample_config.yaml")
    gripper = RobotGripper.load("./config/gripper_params.yaml")

    cloud = None
    cloud_voxel = None
    if DATASET == "fusion":  # Note
        cloud = o3d.io.read_point_cloud(file_dir + "/surface_cloud_with_normals.pcd")
        cloud_voxel = o3d.io.read_point_cloud(file_dir + "/surface_cloud_voxel.pcd")
    elif DATASET == "ycb":
        cloud = o3d.io.read_point_cloud(file_dir + "/nontextured.ply")
        cloud_voxel_file = file_dir + "/nontextured_voxel.ply"
        if not os.path.exists(cloud_voxel_file):
            cloud_voxel = o3d.geometry.voxel_down_sample(cloud, voxel_size=0.008)
            o3d.io.write_point_cloud(cloud_voxel_file, cloud_voxel)
        else:
            cloud_voxel = o3d.io.read_point_cloud(cloud_voxel_file)

    obj = GraspableObject(cloud, cloud_voxel)

    test_grasp_sample()
