from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
from models.config import config as cfg


def main():
    dataset = np.load("../data/m30/pc_train.npy")
    np.random.shuffle(dataset)
    for i in range(len(dataset)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        visualize_pc(dataset[i], ax)
        plt.show(fig)
        plt.close(fig)
        
#        cmd = cv2.waitkey(10) % 256
#        if cmd == ord('q'):
#            break
#        elif cmd == ord('n'):
#            continue

    return


def visualize_pc(pc, ax):
    voxel_coords = ((pc - np.array([cfg.xrange[0], cfg.yrange[0], cfg.zrange[0]])) /
                    (cfg.vw, cfg.vh, cfg.vd)).astype(np.int32)

    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                    return_inverse=True, return_counts=True)

    for i in range(len(voxel_coords)):
        idx = (inv_ind == i)
        color = (voxel_coords[i][0]/cfg.D, voxel_coords[i][1]/cfg.H, voxel_coords[i][2]/cfg.W)
        ax.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2], c=color)


if __name__ == "__main__":
    main()

