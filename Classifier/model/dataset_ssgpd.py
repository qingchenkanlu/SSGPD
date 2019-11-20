import os
import sys
import glob

import numpy as np
import open3d as o3d  # should import befor torch: https://github.com/intel-isl/Open3D/pull/1262
import torch
import torch.utils.data
import torch.nn as nn

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
from Generator.utils.grasps_save_read import grasps_read

curr_path = os.path.dirname(os.path.abspath(__file__))
# print("curr_path", curr_path)
sample_config = YamlConfig(curr_path + "/../../Generator/config/sample_config.yaml")
gripper = RobotGripper.load(curr_path + "/../../Generator/config/gripper_params.yaml")
ags = GpgGraspSampler(gripper, sample_config)


class PointGraspOneViewDataset(torch.utils.data.Dataset):
    def __init__(self, grasp_points_num, grasp_amount_per_obj, thresh_good,
                 thresh_bad, path, tag, min_point_limit=150, with_obj=False):
        self.grasp_points_num = grasp_points_num
        self.grasp_amount_per_obj = grasp_amount_per_obj
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = min_point_limit  # 最低点数限制

        fl_grasp = []
        for dirpath, dirnames, files in os.walk(path):
            for f in files:
                if f.endswith('.pickle'):
                    fl_grasp.append(os.path.join(dirpath, f))
        print("fl_grasp", fl_grasp)

        self.d_grasp, self.d_pc = {}, {}
        for i in fl_grasp:
            obj_name = os.path.basename(os.path.dirname(i))
            print("obj_name", obj_name)
            pc_dir = os.path.join(os.path.dirname(i), 'processed/clouds')
            print("pc_dir", pc_dir)
            fl_pc = glob.glob(os.path.join(pc_dir, '*.pcd'))
            # print("fl_pc", fl_pc)
            if len(fl_pc) != 0:
                # 获取各物品抓取姿态列表
                self.d_grasp[obj_name] = i
                # 获取各物品点云列表
                self.d_pc[obj_name] = fl_pc
            else:
                print("[ERROR] Object:%s don't have point clouds." % obj_name)

        print("d_grasp", self.d_grasp)
        # print("d_pc", self.d_pc)

        self.object = list(self.d_grasp.keys())
        print("objects to deal with", self.object)
        self.amount = len(self.object) * self.grasp_amount_per_obj

    def collect_pc(self, grasp, pc, label):
        """
        获取手抓闭合区域中的点云
        :param grasp: 扫描仪获取的mesh坐标系下抓取姿态 (grasp_center, grasp_axis, grasp_angle, grasp_width, jaw_width)
        :param pc: 点云
        :param label: 标签
        :return: 手抓闭合区域中的点云
        """
        center, normal, major_pc, minor_pc = grasp.center, grasp.normal, grasp.major_pc, grasp.minor_pc
        bottom_center = -ags.gripper.hand_depth * normal + center

        # NOTE: c:center bc:bottom center p:point cloud
        matrix_p2bc = np.array([normal, major_pc, minor_pc])  # 旋转矩阵: 点云坐标系->底部中心点坐标系
        pc_p2bc = (np.dot(matrix_p2bc, (pc-bottom_center).T)).T  # 原始坐标系下点云转换到中心点坐标系下

        if False:  # NOTE：此处获得的抓取姿态可能与点云存在碰撞(影响不是很大)！！！ TODO：碰撞检查
            mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
            mlab.pipeline.surface(mlab.pipeline.open("/home/sdhm/Projects/SSGPD/Dataset/fusion/2018-04-24-15-57-16/processed/mesh.ply"))
            ags.show_origin()
            ags.show_points(pc, color='lb')
            ags.show_points(pc_p2bc, color='y')
            ags.show_grasp_norm_oneside(bottom_center, normal, major_pc, minor_pc, scale_factor=0.001)
            hand_points = ags.get_hand_points(bottom_center, normal, major_pc)
            ags.show_grasp_3d(hand_points, color='g')
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
            # print("[INFO] points num", len(self.in_ind))
            return None

        if False:  # 显示手抓闭合区域内点云
            mlab.figure(bgcolor=(1, 1, 1), size=(1000, 800))
            mlab.pipeline.surface(mlab.pipeline.open("/home/sdhm/Projects/SSGPD/Dataset/fusion/2018-04-24-15-57-16/processed/mesh.ply"))
            ags.show_origin()

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

    def __getitem__(self, index):
        # 获取物体和抓取姿态索引
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_obj))

        obj_name = self.object[obj_ind]  # 物体名称, 用于获取抓取姿态

        f_grasp = self.d_grasp[obj_name]  # 抓取姿态文件名
        fl_pc = np.array(self.d_pc[obj_name])  # 各视角点云文件名
        np.random.shuffle(fl_pc)  # 打乱文件

        # 随机载入抓取姿态
        grasp = grasps_read(f_grasp)[grasp_ind]
        # 随机载入一个点云
        pc = o3d.io.read_point_cloud(fl_pc[-1])
        pc = np.asarray(pc.points)

        # 根据score分类
        score = grasp[1]
        if score >= self.thresh_bad:
            label = 0
            # print("label: 0")
        elif score <= self.thresh_good:
            label = 1
            # print("label: 1")
        else:
            return None

        # NOTE:获取手抓闭合区域中的点云
        grasp_pc = self.collect_pc(grasp[0], pc, label)

        if grasp_pc is None:
            return None

        # 点数不够则有放回采样, 点数太多则随机采样
        if len(grasp_pc) > self.grasp_points_num:
            grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num, replace=False)].T
        else:
            grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num, replace=True)].T

        if self.with_obj:
            return grasp_pc, label, obj_name
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount


if __name__ == '__main__':

    def worker_init_fn(pid):  # After creating the workers, each worker has an independent seed
        np.random.seed(torch.initial_seed() % (2 ** 31 - 1))


    def my_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


    grasp_points_num = 1000
    thresh_good = 0.6
    thresh_bad = 0.6

    b = PointGraspOneViewDataset(
        grasp_points_num=grasp_points_num,
        path="../../Dataset/fusion",
        tag='train',
        grasp_amount_per_obj=50,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        min_point_limit=150,
    )

    cnt = 0
    for i in range(b.__len__()):
        ret = b.__getitem__(i)
        # print(ret)
        if ret is not None:
            print("[debug] grasp_pc", ret[0], ret[0].shape)
            cnt += 1

    print("[INFO] have {} valid grasp in the dataset({}).".format(cnt, b.__len__()))

    # train_loader = torch.utils.data.DataLoader(
    #     PointGraspOneViewDataset(
    #         grasp_points_num=grasp_points_num,
    #         path=curr_path + "/../../Dataset/fusion",
    #         tag='train',
    #         grasp_amount_per_obj=3200,
    #         thresh_good=thresh_good,
    #         thresh_bad=thresh_bad,
    #         min_point_limit=150,
    #     ),
    #     batch_size=64,
    #     num_workers=32,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=worker_init_fn,
    #     collate_fn=my_collate,
    #     drop_last=True,  # fix bug: ValueError: Expected more than 1 value per channel when training
    # )
    #
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print("data", data, data.shape, "target", target)
    #     pass
