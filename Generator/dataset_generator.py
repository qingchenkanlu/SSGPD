#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : MrRen-sdhm
# File Name  : dataset_generator.py
# Get cloud in grasp hand closed area and label cloud by score.

import os
import glob
import time
import pickle

import numpy as np
import open3d as o3d  # should import befor torch: https://github.com/intel-isl/Open3D/pull/1262

try:
    import mayavi.mlab as mlab
except:
    try:
        import mayavi.mlab as mlab
    except ImportError:
        mlab = []
        print("[ERROR] Failed to import mayavi")

# global configurations:
# sys.path.append("../../Generator")
from autolab_core import YamlConfig
from Generator.grasp.grasp_sampler import GpgGraspSampler
from Generator.grasp.gripper import RobotGripper
from Generator.utils.grasps_show import grasps_read

curr_path = os.path.dirname(os.path.abspath(__file__))
# print("curr_path", curr_path)
sample_config = YamlConfig(curr_path + "/config/sample_config.yaml")
gripper = RobotGripper.load(curr_path + "/config/gripper_params.yaml")
ags = GpgGraspSampler(gripper, sample_config)


class OneViewDatasetGenerator:
    def __init__(self, grasps_file, pc_amount_per_grasp, min_point_limit=150):
        self.pc_amount_per_grasp = pc_amount_per_grasp
        self.grasps_file = grasps_file
        self.min_point_limit = min_point_limit  # 最低点数限制

        fl_grasp = self.grasps_file
        grasps = grasps_read(fl_grasp)
        print("fl_grasp", fl_grasp)
        print("grasp num:%d" % len(grasps))

        obj_name = os.path.basename(os.path.dirname(fl_grasp))
        print("obj_name", obj_name)
        pc_dir = os.path.join(os.path.dirname(fl_grasp), 'processed/clouds')
        print("pc_dir", pc_dir)
        fl_pcs = glob.glob(os.path.join(pc_dir, '*.pcd'))
        print("pointcloud num:%d" % len(fl_pcs))
        # print("fl_pcs", fl_pcs)

        print("objects to deal with", obj_name)
        self.min_data_num = len(grasps)
        self.max_data_num = len(fl_pcs) * len(grasps)
        print("min_data_num:%d max_data_num:%d" % (self.min_data_num, self.max_data_num))
        self.obj_name = obj_name
        self.fl_pcs = fl_pcs
        self.grasps = grasps
        self.data_amount = len(grasps) * pc_amount_per_grasp
        print("data num to generate:%d" % self.data_amount)

    def collect_pc(self, grasp, pc, label, vis=False):
        """
        获取手抓闭合区域中的点云
        :param grasp: 扫描仪获取的mesh坐标系下抓取姿态 (grasp_center, grasp_axis, grasp_angle, grasp_width, jaw_width)
        :param pc: 点云
        :param label: 标签
        :param vis: 可视化开关
        :return: 手抓闭合区域中的点云
        """
        center, normal, major_pc, minor_pc = grasp.center, grasp.normal, grasp.major_pc, grasp.minor_pc
        bottom_center = -ags.gripper.hand_depth * normal + center

        # NOTE: c:center bc:bottom center p:point cloud
        matrix_p2bc = np.array([normal, major_pc, minor_pc])  # 旋转矩阵: 点云坐标系->底部中心点坐标系
        pc_p2bc = (np.dot(matrix_p2bc, (pc-bottom_center).T)).T  # 原始坐标系下点云转换到中心点坐标系下

        if False:
            mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
            mlab.pipeline.surface(mlab.pipeline.open(obj_path + "/processed/mesh.ply"))
            ags.show_origin()
            ags.show_points(pc, color='lb')
            ags.show_grasp_norm_oneside(bottom_center, normal, major_pc, minor_pc, scale_factor=0.001)
            hand_points = ags.get_hand_points(bottom_center, normal, major_pc)
            ags.show_grasp_3d(hand_points, color='g')

            ags.show_points(pc_p2bc, color='b')
            hand_points = (np.dot(matrix_p2bc, (hand_points - bottom_center).T)).T  # 手抓关键点转换到中心点坐标系
            ags.show_grasp_3d(hand_points, color='y')

            mlab.title(str(label), size=0.3, color=(0, 0, 0))
            mlab.show()

        # 获取手抓闭合区域中的点
        x_limit = ags.gripper.hand_depth
        z_limit = ags.gripper.hand_height
        y_limit = ags.gripper.max_width

        x1 = pc_p2bc[:, 0] > 0
        x2 = pc_p2bc[:, 0] < x_limit
        y1 = pc_p2bc[:, 1] > -y_limit/2
        y2 = pc_p2bc[:, 1] < y_limit/2
        z1 = pc_p2bc[:, 2] > -z_limit/2
        z2 = pc_p2bc[:, 2] < z_limit/2

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]  # 手抓闭合区域中点的索引

        if len(self.in_ind) < self.min_point_limit:  # 手抓闭合区域内点数太少
            if False:  # 显示闭合区域点数太少的抓取姿态
                print("[INFO] points num", len(self.in_ind))

                mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
                mlab.pipeline.surface(mlab.pipeline.open(obj_path + "/processed/mesh.ply"))
                ags.show_origin(0.03)

                # 显示原始坐标系下点云及手抓
                ags.show_points(pc, color='lb')
                ags.show_grasp_norm_oneside(bottom_center, normal, major_pc, minor_pc, scale_factor=0.001)
                hand_points = ags.get_hand_points(bottom_center, normal, major_pc)
                ags.show_grasp_3d(hand_points, color='g')

                pc_c2m_region = (np.dot(matrix_p2bc.T, pc_p2bc[self.in_ind].T)).T + bottom_center  # 扫描仪坐标系下手抓闭合区域中的点云
                ags.show_points(pc_c2m_region, color='r', scale_factor=.002)

                # 显示底部中心点坐标系下点云及手抓(应在世界坐标系原点)
                ags.show_points(pc_p2bc, color='b')
                ags.show_points(pc_p2bc[self.in_ind], color='r', scale_factor=.002)  # 中心点坐标系下手抓闭合区域中的点云
                hand_points = (np.dot(matrix_p2bc, (hand_points - bottom_center).T)).T  # 手抓关键点转换到中心点坐标系
                ags.show_grasp_3d(hand_points, color='y')

                mlab.title(str(label), size=0.3, color=(0, 0, 0))
                mlab.show()
            return None

        if False:  # 显示手抓闭合区域内点云
            mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
            mlab.pipeline.surface(mlab.pipeline.open(obj_path + "/processed/mesh.ply"))
            ags.show_origin(0.03)

            # 显示原始坐标系下点云及手抓
            ags.show_points(pc, color='lb')
            ags.show_grasp_norm_oneside(bottom_center, normal, major_pc, minor_pc, scale_factor=0.001)
            hand_points = ags.get_hand_points(bottom_center, normal, major_pc)
            ags.show_grasp_3d(hand_points, color='g')

            pc_c2m_region = (np.dot(matrix_p2bc.T, pc_p2bc[self.in_ind].T)).T + bottom_center  # 扫描仪坐标系下手抓闭合区域中的点云
            ags.show_points(pc_c2m_region, color='r', scale_factor=.002)

            # 显示底部中心点坐标系下点云及手抓(应在世界坐标系原点)
            ags.show_points(pc_p2bc, color='b')
            ags.show_points(pc_p2bc[self.in_ind], color='r', scale_factor=.002)  # 中心点坐标系下手抓闭合区域中的点云
            hand_points = (np.dot(matrix_p2bc, (hand_points - bottom_center).T)).T  # 手抓关键点转换到中心点坐标系
            ags.show_grasp_3d(hand_points, color='y')

            # 显示手抓闭合区域
            # x_arr = np.array([-1, 1, 1, -1, -1, 1, 1, -1])/2
            # y_arr = np.array([-1, -1, 1, 1, -1, -1, 1, 1])/2
            # z_arr = np.array([-1, -1, -1, -1, 1, 1, 1, 1])/2
            # x = (x_arr + 0.5) * ags.gripper.hand_depth  # 平移半个单位
            # y = y_arr * (ags.gripper.hand_outer_diameter-2*ags.gripper.finger_width)
            # z = z_arr * ags.gripper.hand_height
            # triangles = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (1, 5, 6), (1, 2, 6),
            #              (0, 4, 7), (0, 3, 7), (2, 3, 6), (3, 6, 7), (0, 1, 5), (0, 4, 5)]
            # mlab.triangular_mesh(x, y, z, triangles, color=(1, 0, 1), opacity=0.2)

            mlab.title("label:{}  point num:{}".format(label, len(self.in_ind)), size=0.25, color=(0, 0, 0))
            mlab.show()

        return pc_p2bc[self.in_ind]  # 返回手抓闭合区域中的点云(手抓底部坐标系下)

    def generate_dataset(self):
        fl_pcs = np.array(self.fl_pcs)  # 各视角点云文件名
        score_ls = []
        for grasp in self.grasps:
            score = grasp[1]
            if score not in score_ls:
                score_ls.append(score)

        # 每个score对应固定数量的grasp, 每个grasp对应固定数量的点云
        grasp_num_per_score = int(len(self.grasps) / len(score_ls))
        data_num_per_score = self.pc_amount_per_grasp * grasp_num_per_score

        data_num = data_num_per_score * len(score_ls)
        dataset_file_path = obj_path + '/grasps_with_score_%s.npy' % str(data_num)
        if os.path.exists(dataset_file_path):
            print("[WARN] dataset file %s exist" % dataset_file_path)
            return
        else:
            print("[INFO] generate dataset for %s" % self.grasps_file)

        print("[DEBUG] score_ls:", score_ls)
        print("[DEBUG] score num:", len(score_ls))
        print("[DEBUG] pc amount per grasp:", self.pc_amount_per_grasp)
        print("[DEBUG] grasp num per score:", grasp_num_per_score)
        print("[INFO] data num per score:", data_num_per_score)
        print("[INFO] will generate data num:", data_num)

        def generate_data_per_grasp():
            dataset_tmp = []
            for grasp in self.grasps:  # 每个抓取姿态随机生成一个抓取数据
                np.random.shuffle(fl_pcs)  # 打乱文件
                # print("[DEBUG] selected fl_pc:", fl_pcs[-1])
                pc = o3d.io.read_point_cloud(fl_pcs[-1])
                pc = np.asarray(pc.points)

                score = grasp[1]
                # NOTE:获取手抓闭合区域中的点云
                grasp_pc = self.collect_pc(grasp[0], pc, score, vis=False)

                if grasp_pc is not None:  # Note: 数据缩减, 须后续处理以为每个score创建固定数量的抓取数据
                    dataset_tmp.append([grasp_pc, score])

            return dataset_tmp

        # 为每个score创建data_num_per_score个抓取数据
        dataset = []
        count_per_score = np.zeros(len(score_ls), dtype=int)
        while np.sum(count_per_score < data_num_per_score) != 0:
            dataset_tmp = generate_data_per_grasp()
            for data in dataset_tmp:
                for ind, score in enumerate(score_ls):
                    if count_per_score[ind - 1] < data_num_per_score and score == data[1]:
                        dataset.append(data)
                        count_per_score[ind - 1] += 1
            print("count per score:", count_per_score)

        # 验证各score对应抓取数据量
        # score_dic = {}
        # for data in dataset:
        #     score = str(data[1])
        #     if score not in score_dic:
        #         score_dic[score] = 1
        #     else:
        #         score_dic[score] += 1
        # print("score_dic:", score_dic)

        dataset = np.array(dataset)  # [[grasp_pc1, score1], [grasp_pc2, score2] ... [grasp_pcn, scoren]]
        print("[INFO] Generated dataset with %d pair of cloud and score." % len(dataset))
        np.save(dataset_file_path, dataset)
        return data_num

    def __len__(self):
        return self.data_amount


if __name__ == '__main__':
    config = {'pc_amount_per_grasp': 8, 'min_point_limit': 500, 'test_data_percent': 0.2}

    start_time = time.time()
    dataset_path = "../Dataset/fusion"
    fl_grasps = []
    for dirpath, dirnames, files in os.walk(dataset_path):
        for f in files:
            if f.endswith('.pickle'):
                fl_grasps.append(os.path.join(dirpath, f))
    print("fl_grasps", fl_grasps)

    ''' Step1: 为每个物品创建抓取数据, 包含点云以及score, 生成grasps_with_score*.npy文件 '''
    print("[MARK] =======  Step1: generate dataset for every object  ======")
    data_num_generated = 0
    for fl_grasp in fl_grasps:
        obj_path = os.path.dirname(fl_grasp)
        dataset = OneViewDatasetGenerator(fl_grasp, config['pc_amount_per_grasp'], config['min_point_limit'])
        data_num = dataset.generate_dataset()
        data_num_generated += data_num
    print("[INFO] gererated %d data in %ds" % (data_num_generated, time.time()-start_time))

    ''' Step2: 划分数据集, 每个物品的20%作为测试集 '''
    print("[MARK] =======  Step2: split dataset as train and test set  ======")
    test_data_percent = config['test_data_percent']
    thresh_good = 0.45
    thresh_bad = 0.75

    train_data_num = 0
    test_data_num = 0
    for fl_grasp in fl_grasps:  # 每个物品对应一个抓取姿态文件, 遍历各个物品
        obj_path = os.path.dirname(fl_grasp)
        dataset_path = glob.glob(obj_path + "/grasps_with_score*.npy")[0]
        print("[INFO] split dataset:", dataset_path)

        dataset = np.load(dataset_path, allow_pickle=True)
        # print(dataset)

        # 统计各类别抓取数据量
        bad_cnt, good_cnt, mid_cnt = 0, 0, 0
        dataset_good = []
        dataset_bad = []
        for data in dataset:
            score = data[1]
            if score >= thresh_bad:
                dataset_bad.append(data)
                data[1] = 1  # 分配标签 bad
                bad_cnt += 1
            elif score <= thresh_good:
                dataset_good.append(data)
                data[1] = 0  # 分配标签 good
                good_cnt += 1
            else:
                mid_cnt += 1
        min_data_num = min(bad_cnt, good_cnt)
        print("[DEBUG] bad_cnt:%d good_cnt:%d mid_cnt:%d min:%d" % (bad_cnt, good_cnt, mid_cnt, min_data_num))

        # 根据数据量最小的划分数据集, 使得各类别数据量接近
        dataset_bad = np.array(dataset_bad)
        dataset_good = np.array(dataset_good)
        np.random.shuffle(dataset_good)  # 打乱文件
        np.random.shuffle(dataset_bad)  # 打乱文件
        dataset_bad = dataset_bad[:min_data_num]
        dataset_good = dataset_good[:min_data_num]
        print("dataset len", len(dataset_bad), len(dataset_good))

        # 各类别取出20%作为测试集
        test_data_num = int(min_data_num * test_data_percent)
        train_datanum = int(min_data_num - test_data_num)
        print("data_num: %d test_data_num: %d train_datanum: %d" % (min_data_num, test_data_num, train_datanum))
        dataset_bad_train = dataset_bad[:train_datanum]
        dataset_good_train = dataset_good[:train_datanum]
        dataset_bad_test = dataset_bad[train_datanum:]
        dataset_good_test = dataset_good[train_datanum:]
        print("dataset_bad_train num:%d dataset_good_train num:%d" % (len(dataset_bad_train), len(dataset_good_train)))
        print("dataset_bad_test num:%d dataset_good_test num:%d" % (len(dataset_bad_test), len(dataset_good_test)))

        dataset_train = np.concatenate((dataset_bad_train, dataset_good_train), axis=0)
        dataset_test = np.concatenate((dataset_bad_test, dataset_good_test), axis=0)
        print("dataset_train num:%d dataset_test num:%d" % (len(dataset_train), len(dataset_test)))

        # 检查数据分布
        # bad_num = good_num = 0
        # for data in dataset_train:
        #     if data[1] == 0:
        #         bad_num += 1
        #     elif data[1] == 1:
        #         good_num += 1
        # print("bad_num: %d good_num: %d in dataset_train" % (bad_num, good_num))

        # 保存数据集文件
        np.save(obj_path + "/dataset_train_%s.npy" % str(len(dataset_train)), dataset_train)
        np.save(obj_path + "/dataset_test_%s.npy" % str(len(dataset_test)), dataset_test)
        test_data_num += len(dataset_test)
        train_data_num += len(dataset_train)
        # exit()

        print("[INFO] gererated %d train data and %d test data for %d obj in %ds" % (
                                                train_data_num, test_data_num, len(fl_grasps), time.time()-start_time))
