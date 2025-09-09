import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from traj_vae import create_model
from parameter import hyper_parameter
import torch.nn as nn
import torch
import argparse
from torch import optim
from traj_dataset import load_data, load_train_eval
from traj_helper import save_model, print_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from tensorboardX import SummaryWriter
from vis_helper import save_traj_to_img, generate_compact_traj
import os 
import pickle
def mk_logdir(params):
    path1 = 'result'
    path2 = 'result/{}/ckpt'.format(params.exp_name)
    path3 = 'result/{}/log'.format(params.exp_name)
    path4 = 'result/{}/images'.format(params.exp_name)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)
    if not os.path.exists(path4):
        os.makedirs(path4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_variable_dim', type=int, default=10)
    parser.add_argument('--noise_mode', type=int, default=1)
    parser.add_argument('--train_the_model', type=bool, default=True)
    parser.add_argument('--evaluate_model', type=bool, default=False)
    parser.add_argument('--roll_out_test', type=bool, default=False)
    parser.add_argument('--visualize_data_distribution', type=bool, default=False)
    parser.add_argument('--restore_model', type=bool, default=False)
    parser.add_argument('--restore_epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=80)
    args = parser.parse_args()
    params = hyper_parameter(args)
    # mk_logdir(params)
    # train_dataset, validation_dataset, train_loader, validation_loader = load_train_eval(params,test = params.test)
    # ''' create model '''
    model = create_model(params)

    pwd = os.getcwd()
    #checkpoint = torch.load('/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/June29-dim3-v5/ckpt/99_ckpt')
    checkpoint = torch.load('/home/zhoutong/dec_jan/traj_data_process/result/zt_jan_001/ckpt/15_ckpt')
    model.load_state_dict(checkpoint)
    decoder = model.vae_decoder
    encoder = model.vae_encoder
    print('zt')

    TRAJ_LIBRARY_PATH_EVAL = pwd + '/dataset/test_data/test' 
    data_file_list=[]
    for cur_file in os.listdir(TRAJ_LIBRARY_PATH_EVAL):
        cur_traj = os.path.join(TRAJ_LIBRARY_PATH_EVAL, cur_file)
        data_file_list.append(cur_traj)

    init_list = []
    ori_list = []
    type_list = []


    for traj_file in data_file_list:
        with open(traj_file, 'rb') as file:
            traj_mat = pickle.load(file)
        for key, value in traj_mat.items():
            # # value is a dictionary that has label and trajectory
            # library_key = len(traj_library)
            # traj_library[str(library_key)] = {}
            # traj_library[str(library_key)]['traj_type'] = value['traj_type']
            # traj_library[str(library_key)]['traj_mask_0'] = value['traj_weight'].transpose(1,0)
            # traj_library[str(library_key)]['traj_mask_1'] = value['traj_weight'].transpose(1,0)
            zt  = value['trajectory'].transpose(1,0)
            zt_type = value['traj_type']
            init_list.append(zt[0,:4])
            ori_list.append(zt[1:,:4])
            type_list.append(zt_type)
    init_array = np.array(init_list)
    ori_array = np.array(ori_list)
    type_array = np.array(type_list)

    # latent_points = []
    init_tensor_array = torch.tensor(init_array, dtype=torch.float32).cuda()
    ori_tensor_array = torch.tensor(ori_array, dtype=torch.float32).cuda()
    type_tensor_array = torch.tensor(type_array, dtype=torch.float32).cuda()
    
    mu, log_var = encoder(ori_tensor_array, type_tensor_array)
    z = mu
    z = torch.tanh(z)
    latent_points_tensor = z[0]
    init_states_tensor = init_tensor_array
    # recons_traj = self.vae_decoder(z, init_state)


    # latent_points = []
    # for x in np.arange(-1, 1.1, 0.1):
    #     for y in np.arange(-1, 1.1, 0.1):
    #         for z in np.arange(-1, 1.1, 0.1):
    #             latent_points.append([x, y, z])

    #latent_points_tensor = torch.tensor(latent_points, dtype=torch.float32).cuda()

    # init_states = np.zeros((latent_points_tensor.size(0), 4))  # 生成与潜在点数量相匹配的零数组
    # init_states[:, -1] = 5  # 将最后一列设为5，代表速度 v

    # # 将初始状态数组转换为Tensor
    # init_states_tensor = torch.tensor(init_states, dtype=torch.float32)
    # init_states_tensor = init_states_tensor.cuda()


    # 解码潜在点生成轨迹
    with torch.no_grad():  # 不计算梯度
        decoded_trajectories = decoder(latent_points_tensor, init_states_tensor).cpu().numpy()  # 将生成的轨迹移回CPU
        ori_trajectories = ori_tensor_array.cpu().numpy()

    output_dir = '/home/zhoutong/dec_jan/traj_data_process/result/test_reconstruction2/'
    root_output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    skip = 201

    for i in range(len(decoded_trajectories)):
        # 通过编码器和解码器处理轨迹
        original_trajectory_np = ori_trajectories[i]
        reconstructed_trajectory_np = decoded_trajectories[i]

        # 创建新图形和子图
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 绘制原始轨迹
        axs[0].plot(original_trajectory_np[:, 0], original_trajectory_np[:, 1], 'o-', markersize=4, label='Original')
        axs[0].set_title('Original Trajectory')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_xlim(0, 25)
        axs[0].set_ylim(-15, 15)
        axs[0].grid(True)
        # axs[0].axis('equal')
        axs[0].legend()

        # 绘制重建轨迹
        axs[1].plot(reconstructed_trajectory_np[:, 0], reconstructed_trajectory_np[:, 1], 'o-', markersize=4, label='Reconstructed')
        axs[1].set_title('Reconstructed Trajectory')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_xlim(0, 25)
        axs[1].set_ylim(-15, 15)
        axs[1].grid(True)
        # axs[1].axis('equal')
        axs[1].legend()

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trajectory_comparison_{i:04d}.png'))
        plt.close(fig)  # 关闭图形，以避免内存溢出













    # for i, trajectory in enumerate(decoded_trajectories):
    #     if i % skip ==0:
    #         sub_dir = os.path.join(root_output_dir, f'trajectory_batch_{i//skip:02d}')
    #         os.makedirs(sub_dir, exist_ok=True)
    #     fig, ax = plt.subplots()
    #     #waypoints = trajectory.reshape(-1, 2)  # 调整轨迹的形状
    #     waypoints = trajectory[:,:2]  # 调整轨迹的形状
    #     ax.plot(waypoints[:, 0], waypoints[:, 1], 'o-', label='Trajectory')
        
    #     # 绘制初始状态 (假设第三个值是角度theta，用箭头表示方向)
    #     x, y, theta, v = init_states_tensor[i].cpu().numpy()
    #     dx, dy = np.cos(theta), np.sin(theta)
    #     ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='r', ec='r', label='Initial State')
        
    #     # 注明隐状态
    #     latent_point_str = f'latent: ({latent_points_tensor[i][0]:.2f}, {latent_points_tensor[i][1]:.2f}, {latent_points_tensor[i][2]:.2f})'
    #     plt.text(0.5, 1.01, latent_point_str, ha='center', transform=ax.transAxes)
        
    #     # 可选：添加图例
    #     ax.legend()
    #     plt.xlim(0,25)
    #     plt.ylim(-15,15)

    #     # 保存图像
    #     #plt.savefig(os.path.join(output_dir, f'trajectory_{i:04d}.png'))
    #     plt.savefig(os.path.join(sub_dir, f'trajectory_{i:04d}.png'))
    #     plt.close(fig)  # 关闭图形，以避免内存溢出

    # print(f"Saved {len(decoded_trajectories)} images to {output_dir}")






# # 假设 decoder 是您已经训练好的 VAE 的解码部分
# # decoder = ...
# decoder = decoder.cuda()  # 将解码器移至CUDA

# # 确保以下目录存在或者修改为您希望保存图片的目录
# output_dir = 'vae_trajectories'
# os.makedirs(output_dir, exist_ok=True)

# 生成潜在点
# latent_points = []
# for x in np.arange(-1, 1.1, 0.1):
#     for y in np.arange(-1, 1.1, 0.1):
#         for z in np.arange(-1, 1.1, 0.1):
#             latent_points.append([x, y, z])

# latent_points = torch.tensor(latent_points, dtype=torch.float32).cuda()

# # 解码潜在点生成轨迹
# with torch.no_grad():  # 不计算梯度
#     decoded_trajectories = decoder(latent_points).cpu().numpy()  # 将生成的轨迹移回CPU

# # 假设轨迹有20个waypoints，并且每个waypoint有2个维度
# for i, trajectory in enumerate(decoded_trajectories):
#     fig, ax = plt.subplots()
#     waypoints = trajectory.reshape(-1, 2)  # 调整轨迹的形状
#     ax.plot(waypoints[:, 0], waypoints[:, 1], marker='o')  # 绘制轨迹

#     # 注明隐状态
#     latent_point_str = f'({latent_points[i][0]:.1f}, {latent_points[i][1]:.1f}, {latent_points[i][2]:.1f})'
#     plt.text(0.5, 1.01, latent_point_str, ha='center', transform=ax.transAxes)

#     # 保存图像
#     plt.savefig(os.path.join(output_dir, f'trajectory_{i:04d}.png'))
#     plt.close(fig)  # 关闭图形，以避免内存溢出

# print(f"Saved {len(decoded_trajectories)} images to {output_dir}")


if __name__ == '__main__':
    main()