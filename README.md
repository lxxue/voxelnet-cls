# voxelnet-cls
A dumb implementation of [voxelnet](https://arxiv.org/abs/1711.06396) in pytorch for object classification.

Code mostly based on [voxnel-pytorch](https://github.com/skyhehe123/VoxelNet-pytorch).

Tested on Ubuntu 16.04 with pytorch 0.3.1(0.4.0+ doesn't work).

Data should be saved as .npy file with shape (None, 1024, 3). Labels should be saved as .npy with shape (None,).

Or you can copy the model and script and then modify them based on your needs.
