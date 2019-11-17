#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 20/05/2018 2:45 PM
# File Name  : dataset_generator.py

import os
import numpy as np
import open3d as o3d
from gripper import RobotGripper
from autolab_core import YamlConfig
from quality import PointGraspMetrics3D
from grasp_sampler import GpgGraspSampler
from graspable_object import GraspableObject
from Generator.utils.grasps_save_read import grasps_save

import multiprocessing


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    file_list.sort()
    return file_list


def do_job(job_num):
    object_name = os.path.basename(file_list[job_num])
    good_grasp = multiprocessing.Manager().list()
    p_set = [multiprocessing.Process(target=worker, args=(job_num, worker_num, good_grasp))
             for worker_num in range(sample_config['num_worker_per_job'])]
    [p.start() for p in p_set]
    [p.join() for p in p_set]

    # all jobs done, prepare for data save
    good_grasp = list(good_grasp)
    good_grasp_file_name = os.path.join(file_list[job_num], 'grasps')
    print("[DEBUG] Job %d save good grasps to file:" % job_num, good_grasp_file_name)
    grasps_save(good_grasp, good_grasp_file_name)

    # tmp = []
    # for grasp in good_grasp:
    #     grasp_config = grasp[0].configuration
    #     score_friction = grasp[1]
    #     score_canny = grasp[2]
    #     tmp.append(np.concatenate([grasp_config, [score_friction, score_canny]]))
    # np.save(good_grasp_file_name + '.npy', np.array(tmp))

    print("\n[WARN] Finished job:", job_num, "object:", object_name)


def worker(job_num, worker_num, good_grasp):
    target_num_grasps = sample_config['target_num_grasps']
    max_num_samples = sample_config['max_num_samples']
    grasp_num_per_fc = sample_config['grasp_num_per_fc']

    object_name = os.path.basename(file_list[job_num])
    data_dir = os.path.join(file_dir, object_name, 'processed')
    print('[DEBUG] a worker of object {} start, job num:{}, worker num:{}'.format(object_name, job_num, worker_num))

    ags = GpgGraspSampler(gripper, sample_config)

    cloud = o3d.io.read_point_cloud(data_dir + "/surface_cloud_with_normals.pcd")
    cloud_voxel = o3d.io.read_point_cloud(data_dir + "/surface_cloud_voxel.pcd")

    obj = GraspableObject(cloud, cloud_voxel)

    good_count_perfect = np.zeros(len(fc_list), dtype=int)

    count = 0
    # 各个摩擦系数生成一些抓取
    while np.sum(good_count_perfect < grasp_num_per_fc) != 0:
        print("[INFO] Job %d worker %d good | min" % (job_num, worker_num), good_count_perfect, grasp_num_per_fc)
        # print("[INFO]:good < mini", good_count_perfect < grasp_num_per_fc,
        #       np.sum(good_count_perfect < grasp_num_per_fc))
        print("[INFO] Job %d worker %d sample grasps..." % (job_num, worker_num))
        grasps = ags.sample_grasps(obj, num_grasps=target_num_grasps, max_num_samples=max_num_samples)
        # print("\033[0;32m%s\033[0m" % "[INFO] Worker{} generate {} grasps.".format(i, len(grasps)))
        count += len(grasps)
        for grasp in grasps:  # 遍历生成的抓取姿态, 判断是否为力闭合, 及其对应的摩擦系数
            tmp, is_force_closure = False, False
            contacts_found, contacts = grasp.close_fingers(obj)
            if not contacts_found:
                continue  # 跳过无接触点的抓取
            for ind_, value_fc in enumerate(fc_list):  # 为每个摩擦系数分配抓取姿态
                value_fc = round(value_fc, 2)
                tmp = is_force_closure
                is_force_closure, _ = PointGraspMetrics3D.grasp_quality(grasp, obj, value_fc, contacts=contacts)
                # print("[INFO] is_force_closure:", bool(is_force_closure), "value_fc:", value_fc, "tmp:", tmp)
                if tmp and not is_force_closure:  # 前一个摩擦系数下为力闭合, 当前摩擦系数下非力闭合, 即找到此抓取对应的最小摩擦系数
                    # print("[debug] tmp and not is_force_closure,value_fc:", value_fc, "ind_:", ind_)
                    if good_count_perfect[ind_ - 1] < grasp_num_per_fc:
                        good_count_perfect[ind_ - 1] += 1  # 前一个摩擦系数最小
                        good_grasp.append((grasp, round(fc_list[ind_ - 1], 2)))  # 保存前一个抓取
                        # print("[debug] good_count_perfect[{}] += 1".format(ind_ - 1))
                    break
                elif is_force_closure and np.isclose(value_fc, fc_list[-1]):  # 力闭合并且摩擦系数最小
                    # print("[debug] is_force_closure and value_fc == fc_list[-1]")
                    if good_count_perfect[ind_] < grasp_num_per_fc:
                        good_count_perfect[ind_] += 1  # 已无更小摩擦系数, 此系数最小
                        good_grasp.append((grasp, value_fc))
                    break
    print("[INFO] Job %d worker %d good | min" % (job_num, worker_num), good_count_perfect, grasp_num_per_fc)

    if count == 0:
        good_grasp_rate = 0
    else:
        good_grasp_rate = len(good_grasp) / count
    print('[DEBUG] Job:', job_num, 'Worker:', worker_num, 'Object:', object_name, 'done.\n')


if __name__ == '__main__':
    file_dir = "../Dataset/fusion"
    file_list = get_file_name(file_dir)
    object_numbers = len(file_list)
    print("[file_list]:", file_list, object_numbers, "\n")

    sample_config = YamlConfig("./config/sample_config.yaml")
    gripper = RobotGripper.load("./config/gripper_params.yaml")

    fc_list = [4.0, 3.0, 2.0, 1.7, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
    print("[fc_list]", fc_list)

    job_list = np.arange(object_numbers)
    job_list = list(job_list)
    pool_size = 2  # number of jobs did at same time
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

    # test a worker
    # good_grasp = []
    # worker(0, 20, 10, good_grasp)
    # exit()

