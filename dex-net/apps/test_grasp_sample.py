#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: sdhm

import numpy as np
import sys
import pickle
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os


def test_grasp_sample(target_num_grasps):
    """
    :param target_num_grasps: 抓取姿态生成器每次的目标生成抓取姿态数
    """
    if False:
        # Test AntipodalGraspSampler
        ags = AntipodalGraspSampler(gripper, yaml_config)
        grasps = ags.sample_grasps(obj, target_num_grasps, grasp_gen_mult=10, max_iter=3, vis=False, random_approach_angle=True)
    else:
        # Test GpgGraspSampler
        ags = GpgGraspSampler(gripper, yaml_config)
        grasps = ags.sample_grasps(obj, num_grasps=50, max_num_samples=10, vis=False)

    # test quality
    force_closure_quality_config = {}
    canny_quality_config = {}
    fc_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    good_count_perfect = np.zeros(len(fc_list))
    for grasp in grasps:
        tmp, is_force_closure = False, False
        print("good_count_perfect", good_count_perfect)
        for ind_, value_fc in enumerate(fc_list):  # 为每个摩擦系数分配抓取姿态
            value_fc = round(value_fc, 2)
            tmp = is_force_closure
            yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
            yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc

            force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(yaml_config['metrics']['force_closure'])
            canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(yaml_config['metrics']['robust_ferrari_canny'])

            is_force_closure, contacts_found = PointGraspMetrics3D.grasp_quality(grasp, obj,  # 依据摩擦系数 value_fc 评估抓取姿态
                                                                 force_closure_quality_config[value_fc], vis=False)

            print("[INFO] is_force_closure:", bool(is_force_closure), "value_fc:", value_fc, "tmp:", bool(tmp))
            if tmp and not is_force_closure:  # 前一个摩擦系数下为力闭合, 当前摩擦系数下非力闭合, 即找到此抓取对应的最小摩擦系数
                print("[debug] tmp and not is_force_closure,value_fc:", value_fc, "ind_:", ind_)
                canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj, canny_quality_config[
                    round(fc_list[ind_ - 1], 2)], vis=False)
                good_count_perfect[ind_ - 1] += 1
                # if value_fc < 0.45:
                #     ags.display_grasps3d([grasp], 'g')
                # else:
                #     ags.display_grasps3d([grasp], 'r')
                # break
            elif is_force_closure and value_fc == fc_list[-1]:  # 力闭合并且摩擦系数最小
                print("[debug] is_force_closure and value_fc == fc_list[-1]")
                canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                  canny_quality_config[value_fc], vis=False)
                good_count_perfect[ind_] += 1
                ags.display_grasps3d([grasp], 'b')
                break

            if not contacts_found:
                ags.new_window(800)
                ags.show_surface_points(obj)
                ags.display_grasps3d([grasp], 'r')
                ags.show()

                PointGraspMetrics3D.grasp_quality(grasp, obj,  # 依据摩擦系数 value_fc 评估抓取姿态
                                                  force_closure_quality_config[value_fc], vis=False)
                break

    ags.show_surface_points(obj)
    # ags.show()

    return True


if __name__ == '__main__':
    home_dir = os.environ['HOME']
    file_dir = home_dir + "/Projects/GPD_PointNet/dataset/ycb_meshes_google/003_cracker_box/google_512k"
    yaml_config = YamlConfig(home_dir + "/Projects/GPD_PointNet/dex-net/test/config.yaml")
    gripper = RobotGripper.load('robotiq_85', home_dir + "/Projects/GPD_PointNet/dex-net/data/grippers")

    if os.path.exists(file_dir + "/nontextured.obj"):
        of = ObjFile(file_dir + "/nontextured.obj")
        sf = SdfFile(file_dir + "/nontextured.sdf")
    else:
        print("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")

    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)

    test_grasp_sample(20)
