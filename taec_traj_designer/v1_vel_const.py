import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER

VEL_LIBRARY_PATH = DATA_RAW_FOLDER + '/vel/'
ensure_directory_exists(VEL_LIBRARY_PATH)
print(VEL_LIBRARY_PATH)


mode = 'save'
if mode == 'save':
    # 生成第一个维度
    zt_scale = 10.0
    zt_scale = 3.0
    arr1 = np.arange(0, 3.1, 0.1)

    # 创建第二个维度，全为1的数组
    # arr2 = np.ones(len(arr1), dtype=float) * zt_scale
    arr2 = np.ones(len(arr1), dtype=float) * zt_scale
    # for i in range(31, 41):
    #     arr2[i] = 0.0
    # 使用堆叠函数vstack将两个数组堆叠在一起
    result = np.vstack((arr1, arr2))
    speed_dict = {}
    speed_id = 0
    speed_dict[str(speed_id)] = result
    speed_id +=1

    mat_name = VEL_LIBRARY_PATH + "vel_init_" + str(zt_scale) + ".pkl"
    save_pickle(mat_name, speed_dict)
