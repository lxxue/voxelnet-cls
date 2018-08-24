import numpy as np


class RotatePC(object):

    def __call__(self, data_pair):
        pc, label = data_pair
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

        return rotated_pc, label


class JitterPC(object):

    def __init__(self, sigma, clip):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_pair):
        data, label = data_pair
        n, c = data.shape
        jittered_data = np.clip(self.sigma * np.random.randn(n, c), -1 * self.clip, self.clip)
        data += jittered_data
        return data, label
