import os
import argparse
from os.path import abspath, dirname
import sys
import time
import logging

import torch
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataloader
from torch.autograd import Variable

sys.path.append(dirname(dirname(abspath(__file__))))
from datasets import modelnet
from models import config as cfg
from utils import data_transform
from models import voxelnet


def main():
    torch.backends.cudnn.enable = True
    logger = init_logger()
    transform = T.Compose([data_transform.RotatePC(),
                           data_transform.JitterPC(0.01, 0.05),
                           T.ToTensor()])
    if args.few:
        train_dataset = modelnet.FewModelNet(args.dset_dir, 'train', transform, args.num_class,
                                             num_perclass=9)
    else:
        train_dataset = modelnet.ModelNet(args.dset_dir, 'train', transform)

    val_dataset = modelnet.ModelNet(args.dset_dir, 'test', transform)

    train_loader = Dataloader(train_dataset, batch_size=cfg.N, shuffle=True, num_worker=6, pin_memory=True)
    val_loader = Dataloader(val_dataset, batch_size=cfg.N, shuffle=False, num_worker=6, pin_memory=True)

    net = voxelnet.VoxelNet(args.num_class, input_shape=(16, 16, 16))
    net.cuda()
    optimizer = optim.SGD(net.parameters(), args.lr, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.8)

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(args.max_epoch):
        logger.info("Epoch: {}".format(i+1))
        scheduler.step()
        train_one_epoch(net, train_loader, optimizer, criterion, logger)
        val_one_epoch(net, val_loader, logger)
    return


def train_one_epoch(net, train_loader, optimizer, criterion, logger):
    net.train()

    t0 = time.time()
    correct = 0
    all = 0
    num_batch = len(train_loader)
    total_loss = 0.

    for i, (batch_data, batch_label) in enumerate(train_loader):
        batch_data = Variable(batch_data.cuda())
        batch_label = Variable(batch_label.cuda())

        optimizer.zero_grad()
        score = net(batch_data)
        loss = criterion(score, batch_label)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(score, axis=1)
        correct += batch_label.eq(pred).sum()
        all += len(batch_label)
        total_loss += loss.data

        if i % 10 == 9:
            logger.info("\titer {}/{}: loss {.4f}, train_acc {.4f}".format(i+1, num_batch, total_loss/(i+1), 100.*correct/all))

    t1 = time.time()
    logger.info("\tTimer: {} sec.".format(t1-t0))
    return


def val_one_epoch(net, val_loader, logger):
    net.eval()

    t0 = time.time()
    correct = 0
    all = 0

    for i, (batch_data, batch_label) in enumerate(val_loader):
        batch_data = Variable(batch_data.cuda())
        batch_label = Variable(batch_label.cuda())
        score = net(batch_data)
        _, pred = torch.max(score, axis=1)
        correct += batch_label.eq(pred).sum()
        all += len(batch_label)

    t1 = time.time()
    logger.info("\tTest: val_acc {.4f}".format(100.*correct/all))
    logger.info("\tTimer: {} sec.".format(t1-t0))
    return


def init_logger():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    log_fname = os.path.join(args.log_dir, "log.txt")
    fh = logging.FileHandler(log_fname)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(acstime)s:%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


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
