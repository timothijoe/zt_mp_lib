import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER
from taec_utils.mpc_utils import iterative_linear_mpc_control, calc_nearest_index, State, MAX_TIME, calc_ref_trajectory, update_state, DT, smooth_yaw
import math
TRAJ_LIBRARY_PATH = DATA_RAW_FOLDER + '/traj/'
MPC_LIBRARY_PATH =  DATA_RAW_FOLDER + '/mpc_track_traj/'
MPC_VISUAL_LIBRARY_PATH =  DATA_RAW_FOLDER + '/mpc_visual2/'
ensure_directory_exists(MPC_VISUAL_LIBRARY_PATH)
import copy

show_both_traj = False
# show_both_traj = True

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


def plant_model_batch(prev_state_batch, pedal_batch, steering_batch, dt = 0.1, st_rate_constrain=0.5):
    #import copy
    prev_state = prev_state_batch
    x_t = prev_state[0]
    y_t = prev_state[1]
    psi_t = prev_state[2]
    v_t = prev_state[3]

    beta = steering_batch
    a_t = pedal_batch
    v_t_1 = v_t + a_t * dt 
    psi_dot = v_t * np.tan(beta) / 2.5
    psi_t_1 = psi_dot*dt + psi_t 
    x_dot = v_t_1 * np.cos(psi_t_1)
    y_dot = v_t_1 * np.sin(psi_t_1)
    x_t_1 = x_dot * dt + x_t 
    y_t_1 = y_dot * dt + y_t
    
    #psi_t = self.wrap_angle_rad(psi_t)
    current_state = np.array([x_t_1, y_t_1, psi_t_1, v_t_1])
    #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
    return current_state

def track_trajectories(trajectories, output_folder, filename):
    """
    绘制一个文件中的多个轨迹，并保存为单独的图像。

    参数:
        trajectories (list): 包含轨迹的列表，每个轨迹是一个 NumPy 数组。
        output_folder (str): 保存图像的文件夹路径。
        filename (str): 当前文件的名字，用于命名输出图像。
    """
    # plt.figure()
    plt.figure(figsize=(10, 6)) 
    for i, trajectory_ele in enumerate(trajectories):
        #trajectory_ele = {}
        raw_trajectory = trajectory_ele['raw_trajectory']
        track_trajectory = trajectory_ele['track_trajectory']
        trajectory = track_trajectory
        x = trajectory[:, 0]  # x 坐标
        y = trajectory[:, 1]  # y 坐标
        init_state = trajectory[0][:4]
        last_state = copy.deepcopy(init_state)
        state_list = []
        state_list.append(init_state)
        control_list = trajectory_ele['track_action']
        for i in range(len(control_list)):
            if i == 0:
                continue  
            control = control_list[i]
            my_state = plant_model_batch(last_state, 0, control[1])
            state_list.append(my_state)
            last_state = copy.deepcopy(my_state)
        state_array = np.array(state_list)
        sx = state_array[:, 0]  # x 坐标
        sy = state_array[:, 1]  # y 坐标
        plt.plot(x, y, color='red', linewidth = 3, label=f"Trajectory {i+1}")
        plt.plot(sx, sy, color='green', linewidth = 2, label=f"Trajectory {i+1}")      
            
        # if (not show_both_traj):
        #     plt.plot(x, y, label=f"Trajectory {i+1}")
        #     norm = plt.Normalize(trajectory[:,0].min(), trajectory[:,0].max())
        #     norm_lon = norm(trajectory[:,0])
        #     # print(trajectory[:,:4])
        #     #plt.scatter(trajectory[:,0], trajectory[:,1], c=norm_lon, cmap='viridis')
        #     plt.scatter(trajectory[-20:,0], trajectory[-20:,1], cmap='viridis')
        # else:
        #     ox = raw_trajectory[:, 0]
        #     oy = raw_trajectory[:, 1]
        #     plt.plot(ox, oy, color='red', linewidth = 2, label=f"Trajectory {i+1}")
        #     plt.plot(x, y, color='green', linewidth = 2, label=f"Trajectory {i+1}")
   
    plt.grid()
    plt.axis("equal")  # 保持 x 和 y 轴比例一致     
    ax = plt.gca()
    ax.set_xlim(0, 15)
    ax.set_ylim(-7.5, 7.5)
    image_name = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(output_folder, image_name)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")
        

def main():
    input_path = MPC_LIBRARY_PATH
    folder_path = MPC_VISUAL_LIBRARY_PATH
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