# copy from https://github.com/lxxue/VoxelNet-pytorch/blob/master/voxelnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from collections import OrderedDict
import sys

from .config import config as cfg

# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x, mask):
        # point-wise feature
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, cfg.T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(6, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


# Convolutional Middle Layer
# class CML(nn.Module):
#
#     def __init__(self):
#         super(CML, self).__init__()
#         self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
#         self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
#         self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
#
#     def forward(self, x):
#         x = self.conv3d_1(x)
#         x = self.conv3d_2(x)
#         x = self.conv3d_3(x)
#         return x

# copy from https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
class VoxNet(torch.nn.Module):

    def __init__(self, num_classes, input_shape):
                 # weights_path=None,
                 # load_body_weights=True,
                 # load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.
        Modified in order to accept different input shapes.
        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        weights_path: str or None, optional
            Default: None
        load_body_weights: bool, optional
            Default: True
        load_head_weights: bool, optional
            Default: True
        Notes
        -----
        Weights available at: url to be added
        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNet, self).__init__()
        self.body = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv3d(in_channels=128,
                                      # out_channels=32, kernel_size=5, stride=2)),
                                      out_channels=32, kernel_size=5, stride=2)),
            ('lkrelu1', torch.nn.LeakyReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)),
            ('lkrelu2', torch.nn.LeakyReLU()),
            # ('pool2', torch.nn.MaxPool3d(2)),
            ('pool2', torch.nn.MaxPool3d(2)),# #
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 128) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        print(first_fc_in_features)

        self.head = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(first_fc_in_features, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, num_classes))
        ]))

        # if weights_path is not None:
        #    weights = torch.load(weights_path)
        #    if load_body_weights:
        #        self.body.load_state_dict(weights["body"])
        #    elif load_head_weights:
        #        self.head.load_state_dict(weights["head"])

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class VoxelNet(nn.Module):

    def __init__(self, num_classes, input_shape):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = VoxNet(num_classes, input_shape)

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(dim, cfg.N, cfg.D, cfg.H, cfg.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = sparse_features

        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):

        # feature learning network
        vwfs = self.svfe(voxel_features)
        # print(vwfs.size())
        vwfs = self.voxel_indexing(vwfs, voxel_coords)
        # print(vwfs.size())

        # convolutional middle network
        cml_out = self.cml(vwfs)
        # print(cml_out.size())

        # region proposal network

        # merge the depth and feature dim into one, output probability score map and regression map
        # psm,rm = self.rpn(cml_out.view(cfg.N, -1, cfg.H, cfg.W))

        return cml_out


