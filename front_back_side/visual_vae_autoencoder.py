import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER
from taec_utils.mpc_utils import iterative_linear_mpc_control, calc_nearest_index, State, MAX_TIME, calc_ref_trajectory, update_state, DT, smooth_yaw
import math

from vae_train.traj_vae import VaeEncoder, VaeDecoder
import torch 
TRAJ_LIBRARY_PATH = DATA_RAW_FOLDER + '/traj/'
MPC_LIBRARY_PATH =  DATA_RAW_FOLDER + '/mpc_track_traj/'
MPC_VISUAL_LIBRARY_PATH =  DATA_RAW_FOLDER + '/mpc_visual3/'
ensure_directory_exists(MPC_VISUAL_LIBRARY_PATH)

device = 'cpu'
device = 'cuda'

encoder_state_dict = torch.load('/home/zhoutong/dir_sda/betty/zt_mp_lib/result/zt_jan13_024/ckpt/150_encoder_ckpt',map_location=torch.device(device))
decoder_state_dict = torch.load('/home/zhoutong/dir_sda/betty/zt_mp_lib/result/zt_jan13_024/ckpt/150_decoder_ckpt',map_location=torch.device(device))

vae_encoder = VaeEncoder(
    embedding_dim = 64,
    h_dim = 128,
    latent_dim = 5,
    seq_len = 30,
    dt = 0.1,
    device = device,
)


vae_decoder = VaeDecoder(
    embedding_dim = 64,
    h_dim = 128,
    latent_dim = 5,
    seq_len = 30,
    dt = 0.1,
    device = device,
)

vae_encoder.load_state_dict(encoder_state_dict)
vae_encoder.to(device)
vae_decoder.load_state_dict(decoder_state_dict)
vae_decoder.to(device)


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
        track_trajectory = trajectory_ele['track_trajectory']
        trajectory = track_trajectory
        origin_trajectory = track_trajectory 
        input_trajectory = np.concatenate((trajectory_ele['track_trajectory'][:,:-1], trajectory_ele['track_action']), axis = 1)
        init_state = input_trajectory[0]
        init_state = torch.from_numpy(init_state).to(torch.float32)
        init_state = init_state.unsqueeze(0)
        init_state = init_state.to(device)
        input_for_encoder = input_trajectory[1:]
        input_for_encoder = torch.from_numpy(input_for_encoder).to(torch.float32).unsqueeze(0)
        input_for_encoder = input_for_encoder.to(device)
        with torch.no_grad():
            mu, sigma = vae_encoder(input_for_encoder)
            z = torch.tanh(mu)
            traj = vae_decoder(z, init_state)
        init_state = init_state[:, :4]
        traj = traj[:,:, :4]
        traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1)
        traj = traj[0,:,:2]
        traj_cpu = traj.detach().to('cpu').numpy()

        x = traj_cpu[:, 0]  # x 坐标
        y = traj_cpu[:, 1]  # y 坐标
        if (not show_both_traj):
            plt.plot(x, y, label=f"Trajectory {i+1}")
            norm = plt.Normalize(trajectory[:,0].min(), trajectory[:,0].max())
            norm_lon = norm(trajectory[:,0])
            # print(trajectory[:,:4])
            #plt.scatter(trajectory[:,0], trajectory[:,1], c=norm_lon, cmap='viridis')
            plt.scatter(traj_cpu[-1:,0], traj_cpu[-1:,1], cmap='viridis')
        else:
            ox = origin_trajectory[:, 0]
            oy = origin_trajectory[:, 1]
            plt.plot(ox, oy, color='red', linewidth = 2, label=f"Trajectory {i+1}")
            plt.plot(x, y, color='green', linewidth = 2, label=f"Trajectory {i+1}")
   
    plt.grid()
    plt.axis("equal")  # 保持 x 和 y 轴比例一致     
    ax = plt.gca()
    ax.set_xlim(0, 15)
    ax.set_ylim(-7.5, 7.5)
    image_name = os.path.splitext(filename)[0] + "is_bothtraj" + str(show_both_traj) + ".png"
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