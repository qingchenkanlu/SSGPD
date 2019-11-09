#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: MrRen-sdhm

import os
import time
import numpy as np

import pcl
from dexnet.grasping.quality_pcd import PointGraspMetrics3D
from dexnet.grasping.grasp_sampler_pcd import GpgGraspSamplerPcd
from dexnet.grasping.graspable_object_pcd import GraspableObjectPcd
from dexnet.grasping.grasp_quality_config_pcd import GraspQualityConfigFactory
from dexnet.grasping import RobotGripper
from autolab_core import YamlConfig
from utils.grasps_save_read import grasps_save, grasps_read

import logging
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')


def test_grasp_sample():
    reload = False
    grasps = None
    # Test GpgGraspSamplerPcd
    ags = GpgGraspSamplerPcd(gripper, yaml_config)
    if not reload:
        grasps = ags.sample_grasps(obj, num_grasps=2000, max_num_samples=155)
        grasps_save(grasps, "/home/sdhm/grasps/test_pcd")
        # exit()

    """ load grasp from file """
    if reload:
        grasps = grasps_read('/home/sdhm/grasps/contact_not_found.pickle')

    if False:
        ags.display_grasps3d(grasps)
        ags.show_surface_points(obj, 'r')
        ags.show()

    mark = 0
    # start the time
    start = time.perf_counter()
    # test quality
    force_closure_quality_config = {}
    canny_quality_config = {}
    fc_list = [4.0, 3.0, 2.0, 1.7, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
    good_count_perfect = np.zeros(len(fc_list))
    contacts_not_found_num = 0
    contacts_found_not_force_closure_num = 0
    proccessed_num = 0
    grasps_save_ls = []
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
            yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
            yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc

            force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(yaml_config['metrics']['force_closure'])
            canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(yaml_config['metrics']['robust_ferrari_canny'])

            is_force_closure, _ = PointGraspMetrics3D.grasp_quality(grasp, obj,  # 依据摩擦系数 value_fc 评估抓取姿态
                                                                    force_closure_quality_config[value_fc],
                                                                    contacts=contacts, vis=False)

            # print("[INFO] is_force_closure:", bool(is_force_closure), "value_fc:", value_fc, "tmp:", bool(tmp))
            if tmp and not is_force_closure:  # 前一个摩擦系数下为力闭合, 当前摩擦系数下非力闭合, 即找到此抓取对应的最小摩擦系数
                # print("[DEBUG] tmp and not is_force_closure,value_fc:", value_fc, "ind_:", ind_)
                # canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj, canny_quality_config[round(fc_list[ind_-1], 2)], vis=False)
                good_count_perfect[ind_-1] += 1  # 前一个摩擦系数最小

                # if np.isclose(value_fc, 0.2):
                #     # ags.show_surface_points(obj)
                #     ags.display_grasps3d([grasp], 'g')

                #     ags.show()

                break  # 找到即退出
            elif is_force_closure and np.isclose(value_fc, fc_list[-1]):  # 力闭合并且摩擦系数最小
                # print("[DEBUG] is_force_closure and value_fc == fc_list[-1]")
                # canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj, canny_quality_config[value_fc], vis=False)
                good_count_perfect[ind_] += 1  # 已无更小摩擦系数, 此系数最小

                # ags.display_grasps3d([grasp], 'b')

                break  # 找到即退出

            if not is_force_closure and np.isclose(value_fc, fc_list[-1]):  # 判断结束还未找到对应摩擦系数,并且找到一对接触点
                print("[DEBUG] not is_force_closure but contacts_found")
                contacts_found_not_force_closure_num += 1

                # grasps_save(grasp, "/home/sdhm/grasps/%s" % str(mark))
                # print("[DEBUG] save grasp to %s.pickle" % str(mark))
                # mark += 1

                # ags.display_grasps3d([grasp], 'g')
                # ags.show_surface_points(obj, color='r')
                # ags.show()

            # if not contacts_found:
            #     ags.new_window(800)
            #     ags.show_surface_points(obj)
            #     ags.display_grasps3d([grasp], 'r')
            #     ags.show_points(np.array(grasp.center), color='r', scale_factor=0.005)
            #     ags.show_points(np.array(grasp.endpoints), color='b', scale_factor=0.005)
            #     ags.show()
            #
            #     PointGraspMetrics3D.grasp_quality(grasp, obj,  # 依据摩擦系数 value_fc 评估抓取姿态
            #                                       force_closure_quality_config[value_fc], vis=False)

                break  # 找到即退出

    print("\n\ngood cnt", good_count_perfect)
    print("proccessed grasp num:", len(grasps))
    print("good_count_perfect num:", int(good_count_perfect.sum()))
    print("contacts_not_found num:", contacts_not_found_num)
    print("contacts_found_not_force_closure num:", contacts_found_not_force_closure_num)
    print("classify took {:.2f} s".format(time.perf_counter()-start))

    grasps_save(grasps, "/home/sdhm/grasps/all")
    grasps_save(grasps_save_ls, "/home/sdhm/grasps/contact_not_found")

    # ags.show_surface_points(obj, color='r')
    # ags.show()
    return True


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    file_dir = home_dir + "/Projects/GPD_PointNet/normal_estimator"
    yaml_config = YamlConfig(home_dir + "/Projects/GPD_PointNet/dex-net/test/config.yaml")
    gripper = RobotGripper.load('robotiq_85', home_dir + "/Projects/GPD_PointNet/dex-net/data/grippers")

    if os.path.exists(file_dir + "/cloud.pcd"):
        cloud = pcl.load(file_dir + "/cloud.pcd")
        cloud_voxel = pcl.load(file_dir + "/cloud_voxel.pcd")
        normals = pcl.load(file_dir + "/normals_as_xyz.pcd")  # normals were saved as PointXYZ formate
    else:
        print("can't find any cloud or normals file!")
        raise NameError("can't find any cloud or normals file!")

    obj = GraspableObjectPcd(cloud, normals, cloud_voxel)

    # test_grasp_example()
    test_grasp_sample()
