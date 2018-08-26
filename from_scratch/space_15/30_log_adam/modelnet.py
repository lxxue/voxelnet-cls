import os
import numpy as np
from torch.utils.data import Dataset
import torch


class ModelNet(Dataset):

    def __init__(self, root, dtype, transform=None):
        self.data = np.load(os.path.join(root, "pc_"+dtype+".npy"))
        self.label = np.load(os.path.join(root, "label_"+dtype+".npy"))
        print("loading data: shape ", self.data.shape, self.label.shape)
        assert self.data.shape[0] == self.label.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        assert self.transform is not None
        voxel_features, voxel_coords = self.transform(data)
        # print(voxel_features.type(), voxel_coords.type(), type(label))
        return voxel_features, voxel_coords, torch.from_numpy(label)

    def __len__(self):
        return self.data.shape[0]


class FewModelNet(ModelNet):

    def __init__(self, root, dtype, transform, num_class, num_perclass):
        super().__init__(root, dtype, transform)

        data = []
        label = []
        for i in range(num_class):
            idx = np.squeeze(self.label == i)
            data_i = self.data[idx]
            label_i = self.label[idx]

            randidx = np.random.permutation(len(data_i))[:num_perclass]
            data.append(data_i[randidx])
            label.append(label_i[randidx])

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        print("after random selection: ", self.data.shape, self.label.shape)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        assert self.transform is not None
        # if self.transform is not None:
        voxel_features, voxel_coords = self.transform(data)
        return voxel_features, voxel_coords, label

    def __len__(self):
        return self.data.shape[0]