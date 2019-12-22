# usage:
#     train: python main_1v.py --mode train
#     reload: python main_1v.py --mode train --load-model default_120.model --load-epoch 120
#     tensorboard: tensorboard --logdir ./assets/log/pointnet --port 8080

import argparse
import os
import time
import logging
import open3d as o3d  # should import befor torch: https://github.com/intel-isl/Open3D/pull/1262

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from Classifier.model.dataset_ssgpd import OneViewDatasetLoader
from Classifier.model.pointnet import PointNetCls, DualPointNetCls

# torch.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='SSGPD')
parser.add_argument('--tag', type=str, default='pointnet')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                    help='pre-trained model path')
parser.add_argument('--data-path', type=str, default='../Dataset/fusion', help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=10)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

TIMESTAMP = "{0:%Y_%m%d_%H%M}".format(datetime.now())
logdir = os.path.join('./assets/log/', args.tag, TIMESTAMP)
os.mkdir(logdir)
tensorboard = SummaryWriter(logdir)

logger = logging.getLogger(args.tag)
logger.setLevel(logging.INFO)
if args.mode == 'train':
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler = logging.FileHandler(logdir + '/train_%s.txt' % args.tag)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info(args)

np.random.seed(int(time.time()))


def worker_init_fn(pid):  # After creating the workers, each worker has an independent seed
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))


def my_collate(batch):  # 用于处理__getitem__返回值为None的情况, 选出不是None的返回值
    batch = list(filter(lambda x: x is not None, batch))  # 移除每个batch中值为None的元素
    return torch.utils.data.dataloader.default_collate(batch)


grasp_points_num = 1024
point_channel = 3

train_loader = torch.utils.data.DataLoader(
    OneViewDatasetLoader(
        grasp_points_num=grasp_points_num,
        dataset_path=args.data_path,
        tag='train',
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
    drop_last=True,  # fix bug: ValueError: Expected more than 1 value per channel when training
)

test_loader = torch.utils.data.DataLoader(
    OneViewDatasetLoader(
        grasp_points_num=grasp_points_num,
        dataset_path=args.data_path,
        tag='test',
        max_data_num=None
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
    drop_last=True,
)

# 载入模型
# args.load_model = '/home/sdhm/Projects/SSGPD/Classifier/assets/learned_models/default_160.model'
# args.load_epoch = 160

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

if is_resume or args.mode == 'test':
    model = torch.load(os.path.join(args.model_path, args.load_model), map_location='cuda:{}'.format(args.gpu))
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
else:
    model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2)

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0, 1, 2, 3]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)


def train(model, loader, epoch):
    scheduler.step()
    scheduler.get_lr()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    for batch_idx, (data, target) in enumerate(loader):
        # print("data", data, data.shape, "target", target)
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        # print("data", data, data.shape)

        starttime = time.time()
        output = model(data)
        endtime = time.time()
        # print("forward time:", (endtime - starttime) * 1000 / data.shape[0])

        # print("output", output, output.shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        if batch_idx % args.log_interval == 0:
            log_str = '[INFO] Train Epoch:{} Lr:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(epoch, scheduler.get_lr(),
                                                                                        batch_idx * args.batch_size, len(loader.dataset),
                                                                                        100. * batch_idx * args.batch_size / len(
                                                                                        loader.dataset), loss.item(), args.tag)
            print(log_str)
            logger.info(log_str)
            tensorboard.add_scalar('train_loss', loss.cpu().item(), batch_idx + epoch * len(loader))
    return float(correct) / float(dataset_size)


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0

    # res = []
    # for data, target, obj_name in loader:
    for data, target in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)  # N*C
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        # for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
        #     res.append((i, j[0], k))

    test_loss /= len(loader.dataset)
    acc = float(correct) / float(dataset_size)
    return acc, test_loss


def main():
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if args.mode == 'train':
        for epoch in range(is_resume * args.load_epoch, args.epoch+1):
            acc_train = train(model, train_loader, epoch)
            log_str = '[INFO] Train done, acc={}'.format(acc_train)
            print(log_str)
            logger.info(log_str)
            acc, loss = test(model, test_loader)
            log_str = '[INFO] Test done, acc={}, loss={}'.format(acc, loss)
            print(log_str)
            logger.info(log_str)
            tensorboard.add_scalar('train_acc', acc_train, epoch)
            tensorboard.add_scalar('test_acc', acc, epoch)
            tensorboard.add_scalar('test_loss', loss, epoch)
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model.state_dict(), path)
                print('[MARK] Save model @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('[INFO] Test done, acc={}, loss={}'.format(acc, loss))


if __name__ == "__main__":
    main()
