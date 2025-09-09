import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER
from taec_utils.mpc_utils import iterative_linear_mpc_control, calc_nearest_index, State, MAX_TIME, calc_ref_trajectory, update_state, DT, smooth_yaw
import math
TRAJ_LIBRARY_PATH = DATA_RAW_FOLDER + '/traj/'

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
            trajectories_dict[filename] = [np.array(trajectory) for key, trajectory in data.items()]
    return trajectories_dict


def track_trajectories(trajectories, output_folder, filename):
    """
    绘制一个文件中的多个轨迹，并保存为单独的图像。

    参数:
        trajectories (list): 包含轨迹的列表，每个轨迹是一个 NumPy 数组。
        output_folder (str): 保存图像的文件夹路径。
        filename (str): 当前文件的名字，用于命名输出图像。
    """
    # plt.figure()
    trajectory_list = {}
    for i, trajectory in enumerate(trajectories):
        trajectory_ele = {}
        trajectory_ele['raw_trajectory'] = trajectory 
        if trajectory.shape[1] < 2:
            print(f"Skipping trajectory {i+1} in {filename}: insufficient dimensions.")
            continue
        cx = trajectory[:, 0]  # x 坐标
        cy = trajectory[:, 1]  # y 坐标
        cyaw = trajectory[:, 2]
        cspeed = trajectory[:, 3]
        ctime = trajectory[:, 4]
        
        # plt.plot(cx, cy, color='red', label=f"Trajectory {i+1}")
   
        # plt.grid()
        # plt.axis("equal")  # 保持 x 和 y 轴比例一致     
        # ax = plt.gca()
        # ax.set_xlim(0, 15)
        # ax.set_ylim(-7.5, 7.5)
        # # plt.xlim([0, 15])  # 设置 x
        
        initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=cspeed[0])
        state = initial_state
        if state.yaw - cyaw[0] >= math.pi:
            state.yaw -= math.pi * 2.0
        elif state.yaw - cyaw[0] <= -math.pi:
            state.yaw += math.pi * 2.0
        
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        t = [0.0]
        d = [0.0]
        a = [0.0]

        target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
        print(target_ind)
        ck = None 
        sp = cspeed 
        dl = 1.0
        time = 0.0
        MAX_TIME = 3.0
        odelta, oa = None, None
        cyaw = smooth_yaw(cyaw)

        zt = 1
        while MAX_TIME >= time:
            target_ind = zt
            xref, target_ind, dref = calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl, target_ind)
            zt += 1
            x0 = [state.x, state.y, state.v, state.yaw]  # current state
            oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
                xref, x0, dref, oa, odelta)
            di, ai = 0.0, 0.0
            if odelta is not None:
                di, ai = odelta[0], oa[0]
                state = update_state(state, ai, di)
            time = time + DT
            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            t.append(time)
            d.append(di)
            a.append(ai)
        
        #track_trajectory = np.array([x, y, yaw, v, t]).T 
        track_trajectory = np.column_stack([x, y, yaw, v, t])
        #track_action = np.array([a, d])
        track_action = np.column_stack([a, d])
        trajectory_ele['track_trajectory'] = track_trajectory 
        trajectory_ele['track_action'] = track_action 
        trajectory_list[i] = trajectory_ele
        
        #plt.plot(x, y, color='green', label=f"Trajectory {i+1}")
        
    # print('zt')
    # # plt.plot(x, y, color='green', label=f"Trajectory {i+1}")
    # plt.show()
    image_name = os.path.splitext(filename)[0] + ".png"
    pkl_name = os.path.splitext(filename)[0] + ".pkl"
    output_path = os.path.join(output_folder, pkl_name)
    save_pickle(output_path, trajectory_list)
    # plt.savefig(output_path)
    # plt.close()
    print(f"Saved plot to {output_path}")      
        



def main():
    # 用户输入文件夹路径
    #folder_path = input("Please enter the folder path containing pickle files: ").strip()
    folder_path = '/home/zhoutong/dir_sda/betty/zt_mp_lib/data/raw_data/traj_png2/'
    input_path = '/home/zhoutong/dir_sda/betty/zt_mp_lib/data/raw_data/traj/'
    #input_path = '/home/zhoutong/dir_sda/betty/zt_mp_lib/data/raw_data/path/circle_path/'
    # 检查文件夹是否存在
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