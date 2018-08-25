import os
import argparse
from os.path import abspath, dirname
import sys
import time
import logging
import numpy as np

import torch
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(dirname(dirname(abspath(__file__))))
from datasets import modelnet
from models.config import config as cfg
from utils import data_transform
from models import voxelnet


def main():
    torch.backends.cudnn.enable = True
    logger = init_logger()
    transform = T.Compose([data_transform.RotatePC(),
                           data_transform.JitterPC(0.01, 0.05),
                           data_transform.AppendCenteredCoord(cfg),
                           data_transform.ToTensor()])
    if args.few:
        train_dataset = modelnet.FewModelNet(args.dset_dir, 'train', transform, args.num_class,
                                             num_perclass=9)
    else:
        train_dataset = modelnet.ModelNet(args.dset_dir, 'train', transform)

    val_dataset = modelnet.ModelNet(args.dset_dir, 'test', transform)

    # TODO: fix the problem of last patch
    train_loader = DataLoader(train_dataset, batch_size=cfg.N, shuffle=True,
                              num_workers=6, collate_fn=customized_collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.N, shuffle=False,
                            num_workers=6, collate_fn=customized_collate, pin_memory=True, drop_last=True)

    net = voxelnet.VoxelNet(args.num_class, input_shape=(10, 10, 10))
    net.cuda()
    optimizer = optim.SGD(net.parameters(), args.lr, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.8)

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(args.max_epoch):
        scheduler.step()
        logger.info("Epoch: {} Lr: {}".format(i+1, scheduler.get_lr()[0]))
        train_one_epoch(net, train_loader, optimizer, criterion, logger)
        val_one_epoch(net, val_loader, logger)
    return


def train_one_epoch(net, train_loader, optimizer, criterion, logger):
    net.train()

    t0 = time.time()
    correct = 0
    total_ins = 0
    num_batch = len(train_loader)
    total_loss = 0.

    for i, (voxel_features, voxel_coords, label) in enumerate(train_loader):
        voxel_features = Variable(voxel_features.cuda())
        label = Variable(label.cuda())

        optimizer.zero_grad()
        score = net(voxel_features, voxel_coords)
        loss = criterion(score, label)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(score, dim=1)
        correct += label.eq(pred).sum().data.cpu().numpy()[0]
        total_ins += len(label)
        total_loss += loss.data.cpu().numpy()[0]

        if i % 10 == 9:
            logger.info("\titer {}/{}: loss {:.4f}, train_acc {:.2f}%".format
                        (i+1, num_batch,  total_loss/(i+1), 100.*correct/total_ins))
            # print(type(i), type(num_batch), type(total_loss), type(correct), type(total_ins))
            # logger.info("\titer {}/{}: loss, train_acc ".format(int(i+1), int(num_batch)))

    t1 = time.time()
    logger.info("\tTimer: {:.2f} sec.".format(t1-t0))
    return


def val_one_epoch(net, val_loader, logger):
    net.eval()

    t0 = time.time()
    correct = 0.
    total_ins = 0.

    for i, (voxel_features, voxel_coords, label) in enumerate(val_loader):
        voxel_features = Variable(voxel_features.cuda())
        label = Variable(label.cuda())
        score = net(voxel_features, voxel_coords)
        _, pred = torch.max(score, dim=1)
        correct += label.eq(pred).sum().data.cpu().numpy()[0]
        total_ins += len(label)

    t1 = time.time()
    logger.info("\tTest: val_acc {:.2f}%".format(100. * correct / total_ins))
    logger.info("\tTimer: {:.2f} sec.".format(t1 - t0))
    return


def init_logger():
    os.makedirs(args.log_dir, exist_ok=True)

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    log_fname = os.path.join(args.log_dir, "log.txt")
    fh = logging.FileHandler(log_fname)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def customized_collate(batch):
    voxel_features = []
    voxel_coords = []
    label = []
    for i, sample in enumerate(batch):
        voxel_features.append(sample[0])
        voxel_coords.append(np.pad(sample[1], ((0, 0), (1, 0)),
                                   mode='constant', constant_values=i))
        label.append(sample[2])

    return torch.cat(voxel_features), np.concatenate(voxel_coords), torch.cat(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_dir", type=str, required=True)
    parser.add_argument("--num_class", type=int, required=True)
    parser.add_argument("--max_epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--few", action='store_true')
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    main()
