import os
import sys
import h5py
import numpy as np
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from dataset import *
from pointnet import PointNetCls, DualPointNetCls


def worker_init_fn(pid):  # After creating the workers, each worker has an independent seed
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))


def my_collate(batch):  # 用于处理__getitem__返回值为None的情况, 选出不是None的返回值
    batch = list(filter(lambda x: x is not None, batch))  # 移除每个batch中值为None的元素
    return torch.utils.data.dataloader.default_collate(batch)


use_cuda = False
grasp_points_num = 750
point_channel = 3
model_path = '../assets/learned_models/default_160.model'
save_prefix = '../assets/learned_models/libtorch_'

data_path = '../data'
thresh_good = 0.6
thresh_bad = 0.6


device = torch.device("cuda:0" if use_cuda else "cpu")

model = torch.load(os.path.join(model_path), map_location='cuda:{}'.format(0))
print(model)

model.to(device)

# An example input you would normally provide to your model's forward() method.

example = torch.rand(1, point_channel, grasp_points_num).float()
print(example, example.shape)
if use_cuda:
    example = example.cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
if use_cuda:
    traced_script_module.save(save_prefix + "gpu.pt")
else:
    traced_script_module.save(save_prefix + "cpu.pt")

# test
data_set = PointGraspOneViewDataset(
        grasp_points_num=grasp_points_num,
        path=data_path,
        tag='train',
        grasp_amount_per_file=100,  # 6500
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
    )

# 获取一个点云样本
grasp_pc = None
for i in range(data_set.__len__()):
    try:
        grasp_pc, label = data_set.__getitem__(i)
        print("[debug] grasp_pc", grasp_pc, grasp_pc.shape)
        break
    except (RuntimeError, TypeError, NameError):
        print("[INFO] don't have valid points!")


with torch.no_grad():
    inputs = torch.tensor(grasp_pc).unsqueeze(0).float()
    print(inputs.shape)
    if use_cuda:
        inputs = inputs.to(device)
    output_script = traced_script_module(inputs)
    output = model(inputs)
    print("[debug] output_script", output_script)
    print("[debug] output", output)
    print("[INFO] to libtorch success!")



# test
# test_loader = torch.utils.data.DataLoader(
#     PointGraspOneViewDataset(
#         grasp_points_num=grasp_points_num,
#         path=data_path,
#         tag='train',
#         grasp_amount_per_file=100,  # 6500
#         thresh_good=thresh_good,
#         thresh_bad=thresh_bad,
#     ),
#     batch_size=1,
#     num_workers=32,
#     pin_memory=True,
#     shuffle=True,
#     worker_init_fn=worker_init_fn,
#     collate_fn=my_collate,
#     drop_last=True,
# )

# with torch.no_grad():
#     for data, target in test_loader:
#         inputs, target = data.float(), target.long().squeeze()
#         # print(inputs)
#         # print(inputs.shape)
#         if use_cuda:
#             inputs = inputs.to(device)
#         output_script = traced_script_module(inputs)
#         output = model(inputs)
#         print("[debug] output_script", output_script)
#         print("[debug] output", output)
#         exit("debug")
