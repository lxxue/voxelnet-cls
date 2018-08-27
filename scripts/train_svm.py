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
from models.config_svm import config as cfg
from utils import data_transform
from models import voxelnet_svm


def main():
    torch.backends.cudnn.enable = True
    logger = init_logger()
    logger.info(str(args))
    save_scripts()
    bin_count = voxel_count()
    logger.info("First 30 bins of training data: \n{}".format(str(bin_count[0][:30])))
    logger.info("Vacant percentage: {:2f}%".format(100*(1-np.sum(bin_count[0])/30**3)))
    logger.info("First 30 bins of validation data: \n{}".format(str(bin_count[1][:30])))
    logger.info("Vacant percentage: {:2f}%".format(100*(1-np.sum(bin_count[1])/30**3)))
    transform = T.Compose([data_transform.RotatePC(),
                           data_transform.JitterPC(0.01, 0.05),
                           data_transform.AppendCenteredCoord(cfg)])
                           # To Tensor done in collate function
                           # data_transform.ToTensor()])
    if args.few:
        train_dataset = modelnet.FewModelNet(args.dset_dir, 'train', transform, args.num_class,
                                             num_perclass=9)
    else:
        train_dataset = modelnet.ModelNet(args.dset_dir, 'train', transform)

    val_dataset = modelnet.ModelNet(args.dset_dir, 'test', transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=6, collate_fn=customized_collate, pin_memory=True, drop_last=False)
    # shuffle val_dataset since I drop last batch
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=6, collate_fn=customized_collate, pin_memory=True, drop_last=False)

    net = voxelnet_svm.VoxelNet(args.num_class, input_shape=(cfg.D, cfg.H, cfg.W))
    logger.info("Total # parameters: {}".format(sum([p.numel() for p in net.parameters()])))
    load_checkpoint(logger, net)
    # logger.info("trainable # parameters: {}".format(sum([p.numel() for p in net.parameters() if p.requires_grad])))
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            logger.info("Trainable params {}: {} {}".format(name, param.size(), param.numel()))
        else:
            logger.info("Non-trainable params {}: {} {}".format(name, param.size(), param.numel()))
    net.cuda()
    # optimizer = optim.SGD(net.parameters(), args.lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.8)

    # criterion = torch.nn.CrossEntropyLoss()

    # best_acc = 0.
    # for i in range(args.max_epoch):
    #     scheduler.step()
    #     logger.info("Epoch: {} Lr: {}".format(i+1, scheduler.get_lr()[0]))
    #     train_one_epoch(net, train_loader, optimizer, criterion, logger)
    #     acc = val_one_epoch(net, val_loader, logger, best_acc)
    #     if acc > best_acc:
    #         best_acc = acc
    #         torch.save(net.state_dict(), os.path.join(args.log_dir, "best.pth.tar"))
    #     if i % 5 == 0 or (i+1) % args.max_epoch  == 0:
    #         torch.save(net.state_dict(), os.path.join(args.log_dir, "last.pth.tar"))
    logger.info("Start computing train dset features...")
    train_features, train_labels = get_svm_feature(net, train_dataset, train_loader, logger)
    print(train_features)
    np.save("../data/svm/m10_train_features.npy", train_features)
    np.save("../data/svm/m10_train_labels.npy", train_labels)
    logger.info("Start computing val dset features...")
    val_features, val_labels = get_svm_feature(net, val_dataset, val_loader, logger)
    print(val_features)
    np.save("../data/svm/m10_val_features.npy", val_features)
    np.save("../data/svm/m10_val_labels.npy", val_labels)
    return


def get_svm_feature(net, dataset, loader, logger):
    net.eval()

    t0 = time.time()
    features = np.zeros((len(dataset), 128), dtype=np.float32)
    labels = np.zeros((len(dataset),), dtype=np.int)
    features = torch.from_numpy(features).cuda()
    for i, (voxel_features, voxel_coords, label) in enumerate(loader):
        voxel_features = Variable(voxel_features.cuda())
        voxel_coords = Variable(voxel_coords.cuda())
        # label = Variable(label.cuda())

        features[i] = net(voxel_features, voxel_coords).data
        labels[i] = label

    features = features.cpu().numpy()
    t1 = time.time()
    logger.info("Finish computing all features using {} sec.".format(t1 - t0))
    return features, labels


def train_one_epoch(net, train_loader, optimizer, criterion, logger):
    net.train()

    t0 = time.time()
    correct = 0
    total_ins = 0
    num_batch = len(train_loader)
    total_loss = 0.

    if args.few:
        log_interval = len(train_loader)
    else:
        log_interval = 100 if args.num_class == 30 else 20

    for i, (voxel_features, voxel_coords, label) in enumerate(train_loader):
        voxel_features = Variable(voxel_features.cuda())
        voxel_coords = Variable(voxel_coords.cuda())
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

        if (i + 1) % log_interval == 0:
            logger.info("\t\titer {}/{}: loss {:.4f}, train_acc {:.2f}%".format
                        (i+1, num_batch,  total_loss/log_interval, 100.*correct/total_ins))
            correct = 0
            total_ins = 0
            total_loss = 0.
            # print(type(i), type(num_batch), type(total_loss), type(correct), type(total_ins))
            # logger.info("\titer {}/{}: loss, train_acc ".format(int(i+1), int(num_batch)))

    t1 = time.time()
    logger.info("\t\tTimer: {:.2f} sec.".format(t1-t0))
    return


def val_one_epoch(net, val_loader, logger, best_acc):
    net.eval()

    t0 = time.time()
    correct = 0.
    total_ins = 0.

    for i, (voxel_features, voxel_coords, label) in enumerate(val_loader):
        voxel_features = Variable(voxel_features.cuda())
        voxel_coords = Variable(voxel_coords.cuda())
        label = Variable(label.cuda())
        score = net(voxel_features, voxel_coords)
        _, pred = torch.max(score, dim=1)
        correct += label.eq(pred).sum().data.cpu().numpy()[0]
        total_ins += len(label)

    t1 = time.time()
    acc = 100. * correct / total_ins
    logger.info("\t\tVal: val_acc {:.2f}%/{:.2f}%".format(acc, best_acc))
    logger.info("\t\tTimer: {:.2f} sec.".format(t1 - t0))
    return acc


def init_logger():
    os.makedirs(args.log_dir, exist_ok=False)

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

    voxel_features = torch.from_numpy(np.concatenate(voxel_features))
    # voxel_coords = torch.from_numpy(np.concatenate(voxel_coords).astype(np.long))
    voxel_coords = torch.from_numpy(np.concatenate(voxel_coords).astype(np.long))
    # label = torch.from_numpy(np.concatenate(label))
    label = torch.from_numpy(np.array(label))
    return voxel_features, voxel_coords, label


def voxel_count():
    train_data = np.load(os.path.join(args.dset_dir, "pc_train.npy"))
    val_data = np.load(os.path.join(args.dset_dir, "pc_test.npy"))
    bin_count = [np.zeros((2048,), dtype=np.float), np.zeros((2048,), dtype=np.float)]
    for i, dataset in enumerate([train_data, val_data]):
        for j in range(len(dataset)):
            pc = dataset[j]
            voxel_coords = ((pc - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                            (cfg.vw, cfg.vh, cfg.vd)).astype(np.int32)

            # voxel_coords = voxel_coords[:, [2, 1, 0]]
            _, _, voxel_counts = np.unique(voxel_coords, axis=0,
                                           return_inverse=True, return_counts=True)
            bin_count_j = np.bincount(voxel_counts, minlength=2048)
            bin_count[i] += bin_count_j

    bin_count[0] = bin_count[0] / len(train_data)
    # bin_count[0][0] = cfg.D * cfg.H * cfg.W - np.sum(bin_count[0])
    bin_count[1] = bin_count[1] / len(val_data)
    # bin_count[1][0] = cfg.D * cfg.H * cfg.W - np.sum(bin_count[1])
    return bin_count


def save_scripts():
    os.system("cp {} {}".format("../datasets/modelnet.py", os.path.join(args.log_dir, "modelnet.py")))
    os.system("cp {} {}".format("../models/config.py", os.path.join(args.log_dir, "config.py")))
    os.system("cp {} {}".format("../models/voxelnet.py", os.path.join(args.log_dir, "voxelnet.py")))
    os.system("cp {} {}".format("../utils/data_transform.py", os.path.join(args.log_dir, "data_transform.py")))
    os.system("cp {} {}".format(__file__, os.path.join(args.log_dir, "script.py")))


def load_checkpoint(logger, net):
    logger.info("Loading saved model...")
    assert os.path.isfile(args.ckpt_fname), "Error: no checkpoint file found!"

    old_state_dict = torch.load(args.ckpt_fname)
    new_state_dict = net.state_dict()

    same_weight = {}
    for k, v in old_state_dict.items():
        if k in new_state_dict and new_state_dict[k].size() == old_state_dict[k].size():
            logger.info("same weight: {}".format(k))
            same_weight[k] = v
        else:
            logger.info("discarded weight: {}".format(k))

    new_state_dict.update(same_weight)
    net.load_state_dict(new_state_dict)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_dir", type=str, required=True)
    parser.add_argument("--num_class", type=int, required=True)
    parser.add_argument("--max_epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--few", action='store_true')
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--ckpt_fname", type=str)
    args = parser.parse_args()
    main()
