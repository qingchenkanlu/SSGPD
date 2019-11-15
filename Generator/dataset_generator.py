#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 20/05/2018 2:45 PM
# File Name  : dataset_generator.py

import numpy as np
import sys
import pickle
from Generator.grasp.quality import PointGraspMetrics3D
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os
import multiprocessing
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # for the convenient of run on remote computer


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    file_list.sort()
    return file_list


def do_job(i):
    object_name = file_list_all[i][len(home_dir) + 48:-12]
    good_grasp = multiprocessing.Manager().list()  # 全局列表, 存储 good grasp
    p_set = [multiprocessing.Process(target=worker, args=(i, 100, 20, good_grasp)) for _ in  # 采样点数, 每个摩擦系数对应的抓取姿态个数
             range(1)]  # grasp_amount per friction: 20*40
    [p.start() for p in p_set]
    [p.join() for p in p_set]

    # all jobs done, prepare for data save
    good_grasp = list(good_grasp)
    good_grasp_file_name = "./generated_grasps/{}_{}_{}".format(filename_prefix, str(object_name), str(len(good_grasp)))
    if not os.path.exists('./generated_grasps/'):
        os.mkdir('./generated_grasps/')

    print("\033[0;32m%s\033[0m" % "[INFO] Save good grasp file:" + good_grasp_file_name)
    with open(good_grasp_file_name + '.pickle', 'wb') as f:
        pickle.dump(good_grasp, f)

    tmp = []
    for grasp in good_grasp:
        grasp_config = grasp[0].configuration
        score_friction = grasp[1]
        score_canny = grasp[2]
        tmp.append(np.concatenate([grasp_config, [score_friction, score_canny]]))
    np.save(good_grasp_file_name + '.npy', np.array(tmp))
    print("\nfinished job ", object_name)


def worker(i, target_num_grasps, grasp_num_per_fc, good_grasp):
    """
    :param target_num_grasps: 抓取姿态生成器每次的目标生成抓取姿态数
    :param grasp_num_per_fc: 每个摩擦系数需要生成的抓取姿态数
    """
    object_name = file_list_all[i][len(home_dir) + 48:-12]
    print('[INFO] a worker of task {} start, job num:{}'.format(object_name, i))

    yaml_config = YamlConfig(home_dir + "/Projects/GPD_PointNet/dex-net/test/sample_config.yaml")
    print("yaml_config:", yaml_config.config)

    gripper_name = 'robotiq_85'
    gripper = RobotGripper.load(gripper_name, home_dir + "/Projects/GPD_PointNet/dex-net/data/grippers")
    grasp_sample_method = "gpg"
    if grasp_sample_method == "uniform":
        ags = UniformGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gaussian":
        ags = GaussianGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "antipodal":
        ags = AntipodalGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gpg":
        ags = GpgGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "point":
        ags = PointGraspSampler(gripper, yaml_config)
    else:
        raise NameError("Can't support this sampler")

    print("grasp_sample_method:", grasp_sample_method)

    if os.path.exists(str(file_list_all[i]) + "/nontextured.obj"):
        of = ObjFile(str(file_list_all[i]) + "/nontextured.obj")
        sf = SdfFile(str(file_list_all[i]) + "/nontextured.sdf")
    else:
        print("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    print("[INFO] opened object", i + 1, object_name)

    force_closure_quality_config = {}
    canny_quality_config = {}
    fc_list_sub1 = np.arange(2.0, 0.75, -0.4)
    fc_list_sub2 = np.arange(0.5, 0.36, -0.05)
    fc_list = np.concatenate([fc_list_sub1, fc_list_sub2])  # 摩擦系数列表  fc_list [2.  1.6  1.2  0.8 | 0.5  0.45 0.4 ]
    print("fc_list", fc_list)
    
    for value_fc in fc_list:
        # use multi friction_coef
        value_fc = round(value_fc, 2)
        yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
        yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc

        # create grasp quality config according to multi friction_coef
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['force_closure'])
        canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['robust_ferrari_canny'])

    good_count_perfect = np.zeros(len(fc_list))
    count = 0

    # 各个摩擦系数生成一些抓取
    while np.sum(good_count_perfect < grasp_num_per_fc) != 0 and good_count_perfect[-1] < grasp_num_per_fc:
        print("[INFO]:good | mini", good_count_perfect, grasp_num_per_fc)
        print("[INFO]:good < mini", good_count_perfect < grasp_num_per_fc,
              np.sum(good_count_perfect < grasp_num_per_fc))
        grasps = ags.generate_grasps(obj, target_num_grasps=target_num_grasps, grasp_gen_mult=10,  # 生成抓取姿态
                                     max_iter=3, vis=True, random_approach_angle=True)  # 随机调整抓取方向
        print("\033[0;32m%s\033[0m" % "[INFO] Worker{} generate {} grasps.".format(i, len(grasps)))
        count += len(grasps)
        for grasp in grasps:  # 遍历生成的抓取姿态, 判断是否为力闭合, 及其对应的摩擦系数
            tmp, is_force_closure = False, False
            for ind_, value_fc in enumerate(fc_list):  # 为每个摩擦系数分配抓取姿态
                value_fc = round(value_fc, 2)
                tmp = is_force_closure
                is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj,  # 依据摩擦系数 value_fc 评估抓取姿态
                                                                     force_closure_quality_config[value_fc], vis=False)
                print("[INFO] is_force_closure:", bool(is_force_closure), "value_fc:", value_fc, "tmp:", tmp)
                if tmp and not is_force_closure:  # 前一个摩擦系数下为力闭合, 当前摩擦系数下非力闭合, 即找到此抓取对应的最小摩擦系数
                    print("[debug] tmp and not is_force_closure,value_fc:", value_fc, "ind_:", ind_)
                    if good_count_perfect[ind_ - 1] < grasp_num_per_fc:
                        canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj, canny_quality_config[
                                                                                round(fc_list[ind_ - 1], 2)], vis=False)
                        good_grasp.append((grasp, round(fc_list[ind_ - 1], 2), canny_quality))  # 保存前一个抓取
                        good_count_perfect[ind_ - 1] += 1
                        print("[debug] good_count_perfect[{}] += 1".format(ind_ - 1))
                    break
                elif is_force_closure and np.isclose(value_fc, fc_list[-1]):  # 力闭合并且摩擦系数最小
                    print("[debug] is_force_closure and value_fc == fc_list[-1]")
                    if good_count_perfect[ind_] < grasp_num_per_fc:
                        canny_quality = PointGraspMetrics3D.grasp_quality(grasp, obj,
                                                                          canny_quality_config[value_fc], vis=False)
                        good_grasp.append((grasp, value_fc, canny_quality))
                        good_count_perfect[ind_] += 1
                    break
        print('Worker:', i, 'Object:{} GoodGraspNum:{}\n\n'.format(object_name, good_count_perfect))

    object_name_len = len(object_name)
    object_name_ = str(object_name) + " " * (25 - object_name_len)
    if count == 0:
        good_grasp_rate = 0
    else:
        good_grasp_rate = len(good_grasp) / count
    print('Worker:', i, 'Gripper:{} Object:{} Rate:{:.4f} {}/{}\n\n'.
          format(gripper_name, object_name_, good_grasp_rate, len(good_grasp), count))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename_prefix = sys.argv[1]
    else:
        filename_prefix = "default"
    home_dir = os.environ['HOME']
    file_dir = home_dir + "/Projects/GPD_PointNet/dataset/ycb_meshes_google/"
    file_list_all = get_file_name(file_dir)
    object_numbers = file_list_all.__len__()
    print("[file_list_all]:", file_list_all, object_numbers, "\n")

    # test a worker
    good_grasp = []
    worker(0, 20, 10, good_grasp)
    exit()

    job_list = np.arange(object_numbers)
    job_list = list(job_list)
    pool_size = 1  # number of jobs did at same time
    assert (pool_size <= len(job_list))
    # Initialize pool
    pool = []
    for _ in range(pool_size):
        job_i = job_list.pop(0)
        pool.append(multiprocessing.Process(target=do_job, args=(job_i,)))
    [p.start() for p in pool]
    # refill
    while len(job_list) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                job_i = job_list.pop(0)
                p = multiprocessing.Process(target=do_job, args=(job_i,))
                p.start()
                pool.append(p)
                break
