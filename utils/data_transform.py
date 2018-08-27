import numpy as np
import torch


class RotatePC(object):

    def __call__(self, pc):
        # print(np.max(pc))
        shape_pc = pc.shape
        assert pc.shape[0] == 3 or pc.shape[1] == 3
        assert len(pc.shape) == 2
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        if shape_pc[0] == 3:
            pc = pc.transpose()
            rotated_pc = np.dot(pc, rotation_matrix)
            rotated_pc = rotated_pc.transpose()
        else:
            rotated_pc = np.dot(pc, rotation_matrix)

        # print(np.max(rotated_pc))
        return rotated_pc


class JitterPC(object):

    def __init__(self, sigma, clip):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pc):
        n, c = pc.shape
        jittered_data = np.clip(self.sigma * np.random.randn(n, c), -1 * self.clip, self.clip)
        pc += jittered_data
        # print(np.max(pc))
        return pc


# reference: https://github.com/lxxue/VoxelNet-pytorch/blob/master/data/kitti.py
class AppendCenteredCoord(object):

    def __init__(self, cfg):
        # x, x_step = np.linspace(cfg.xrange[0], cfg.xrange[1], cfg.D, endpoint=False, retstep=True)
        # y, y_step = np.linspace(cfg.yrange[0], cfg.yrange[1], cfg.W, endpoint=False, retstep=True)
        # z, z_step = np.linspace(cfg.zrange[0], cfg.zrange[1], cfg.H, endpoint=False, retstep=True)
        # assert np.allclose([cfg.vd, cfg.vw, cfg.vh], [x_step, y_step, z_step])
        # x, y, z = x + x_step/2, y + y_step/2, z + z_step/2
        # xx, yy, zz = np.meshgrid(x, y, z, sparse=False, indexing='ij')
        # #xx_c, yy_c, zz_c = xx + x_step/2, yy + y_step/2, zz + z_step/2

        # self.voxel_center = np.stack([xx, yy, zz], axis=-1)
        # assert self.voxel_center.shape == (cfg.D, cfg.W, cfg.H, 3)
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        self.vw = cfg.vw
        self.vh = cfg.vh
        self.vd = cfg.vd
        self.W = cfg.W
        self.H = cfg.H
        self.D = cfg.D
        self.T = cfg.T

    def __call__(self, pc):
        voxel_coords = ((pc - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) /
                        (self.vw, self.vh, self.vd)).astype(np.int32)

        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                        return_inverse=True, return_counts=True)
        voxel_features = []
        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 6), dtype=np.float32)
            pts = pc[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T

            voxel[:pts.shape[0], :] = np.concatenate([pts, pts - np.mean(pts, axis=0)], axis=1)
            voxel_features.append(voxel)

        return np.array(voxel_features), voxel_coords


#class ToTensor(object):
#
#    def __call__(self, voxel_stuff):
#        voxel_features, voxel_coords = voxel_stuff
#        return torch.from_numpy(voxel_features), torch.from_numpy(voxel_coords)

