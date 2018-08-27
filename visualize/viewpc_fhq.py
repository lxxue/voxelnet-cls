import numpy as np
from show3d_balls import showpoints

data = np.load("/data/lixin_backup/3DDLComparison/data/modelnet30/pc_train.npy")
label = np.load("/data/lixin_backup/3DDLComparison/data/modelnet30/label_train.npy")

for i in range(len(data)):
    if label[i] == 0:    
        showpoints(data[i])
