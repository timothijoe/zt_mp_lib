import os
import pdb
import glob
import numpy as np
import random
import copy
import scipy.io as io
import pickle
import matplotlib.pyplot as plt

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER
from taec_utils.mpc_utils import iterative_linear_mpc_control, calc_nearest_index, State, MAX_TIME, calc_ref_trajectory, update_state, DT, smooth_yaw
import math
MPC_LIBRARY_PATH =  DATA_RAW_FOLDER + '/mpc_track_traj/'
MPC_STANDARD_LIBRARY_PATH =  DATA_RAW_FOLDER + '/standard_traj/'
ensure_directory_exists(MPC_STANDARD_LIBRARY_PATH)


def load_pickle_files(folder_path):
    """
    从指定文件夹中加载所有 pickle 文件并返回解析后的数据。

    参数:
        folder_path (str): 包含 pickle 文件的文件夹路径。

    返回:
        dict: 包含文件名和解析后的轨迹数据，每个文件的数据是一个列表。
        格式: {filename: [trajectory1, trajectory2, ...]}
    """
    trajectories_dict = {}
    for filename in os.listdir(folder_path):
        # 检查是否是 pickle 文件
        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            file_path = os.path.join(folder_path, filename)
            data = load_pickle(file_path)
            trajectories_dict[filename] = [trajectory for key, trajectory in data.items()]
    return trajectories_dict

def track_trajectories(trajectories, output_folder, filename):
    """
    绘制一个文件中的多个轨迹，并保存为单独的图像。

    参数:
        trajectories (list): 包含轨迹的列表，每个轨迹是一个 NumPy 数组。
        output_folder (str): 保存图像的文件夹路径。
        filename (str): 当前文件的名字，用于命名输出图像。
    """
    weight = 1.0
    if (len(trajectories) == 1):
        weight *= 4
    traj2_library = {}
    for i, trajectory_ele in enumerate(trajectories):
        traj_mask_1 = np.ones((5,30))
        traj_weight = traj_mask_1 * weight
        library_key = i
        traj2_library[library_key] = {}
        traj2_library[library_key]['label'] = 'label'
        traj2_library[library_key]['traj_type'] = 1
        traj2_library[library_key]['traj_weight'] = traj_weight 
        traj2_library[library_key]['raw_trajectory'] = trajectory_ele['raw_trajectory']
        traj2_library[library_key]['track_trajectory'] = trajectory_ele['track_trajectory']
        traj2_library[library_key]['track_action'] = trajectory_ele['track_action']
    if (len(trajectories) == 1):
        import copy
        only_library_ele = copy.deepcopy(traj2_library[0])
        for i in range(10):
            library_key = i + 1
            traj2_library[library_key] = only_library_ele
    pkl_name = os.path.splitext(filename)[0] + ".pkl"
    output_path = os.path.join(output_folder, pkl_name)
    save_pickle(output_path, traj2_library)
    print(f"Saved plot to {output_path}")      
            

def main():
    input_path = MPC_LIBRARY_PATH
    folder_path = MPC_STANDARD_LIBRARY_PATH
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    # 加载轨迹数据
    print("Loading pickle files...")
    trajectories_dict = load_pickle_files(input_path)
    if not trajectories_dict:
        print("No valid pickle files found in the folder.")
        return
    # 创建输出文件夹
    output_folder = os.path.join(folder_path, "output_plots")
    output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)
    # 为每个文件绘制轨迹图
    for filename, trajectories in trajectories_dict.items():
        print(f"Plotting trajectories from {filename}...")
        track_trajectories(trajectories, output_folder, filename)
    print("All plots have been saved.")


# 调用主函数
if __name__ == "__main__":
    main()

