import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io
import copy
import random
import time
import matplotlib.gridspec as gridspec
import cv2
import io
import torch
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def save_traj_to_img(args, origin_traj_lst, generate_traj_lst):
    # input: args: names, including compared two traj name, vel_name
    #        origin_traj_list: the trajectory, batch x seq_len x 3
    #        generate_traj_list

    # output: images that can be sent to tensorboard

    # origin_traj_name = 'trian_expert_traj'
    # generate_traj_name = 'eval_expert_traj'
    # velocity_compare_name = 'train_eval_vel_compare'
    origin_traj_name = args[0]
    generate_traj_name = args[1]
    velocity_compare_name = 'vel_compare'
    file_name = '/traj_' + args[3] + '.jpg'
    file_path = 'result/{}/images'.format(args[2])
    file_name = file_path + file_name
    fig = plt.figure(figsize = (19, 11))
    gs = gridspec.GridSpec(11, 19)
    fig11 = fig.add_subplot(gs[0:3, 0:4])
    fig12 = fig.add_subplot(gs[0:3, 5:9])
    fig13 = fig.add_subplot(gs[0:3, 10:14])
    fig14 = fig.add_subplot(gs[0:3, 15:19])
    fig21 = fig.add_subplot(gs[ 4:7, 0:4])
    fig22 = fig.add_subplot(gs[ 4:7, 5:9])
    fig23 = fig.add_subplot(gs[ 4:7, 10:14])
    fig24 = fig.add_subplot(gs[ 4:7, 15:19])
    fig31 = fig.add_subplot(gs[8:11, 0:4])
    fig32 = fig.add_subplot(gs[8:11, 5:9])
    fig33 = fig.add_subplot(gs[8:11, 10:14])
    fig34 = fig.add_subplot(gs[8:11, 15:19])
    #plt.axis([-1, 10, -6, 6])

    fig11.set_xlim([-1, 20])
    fig11.set_ylim([-6, 6])
    fig12.set_xlim([-1, 20])
    fig12.set_ylim([-6, 6])
    fig13.set_xlim([-1, 20])
    fig13.set_ylim([-6, 6])
    fig14.set_xlim([-1, 20])
    fig14.set_ylim([-6, 6])
    fig21.set_xlim([-1, 20])
    fig21.set_ylim([-6, 6])
    fig22.set_xlim([-1, 20])
    fig22.set_ylim([-6, 6])
    fig23.set_xlim([-1, 20])
    fig23.set_ylim([-6, 6])
    fig24.set_xlim([-1, 20])
    fig24.set_ylim([-6, 6])


    fig31.set_xlim([-0.1, 2.6])
    fig31.set_ylim([-2, 10])
    fig32.set_xlim([-0.1, 2.6])
    fig32.set_ylim([-2, 10])
    fig33.set_xlim([-0.1, 2.6])
    fig33.set_ylim([-2, 10])
    fig34.set_xlim([-0.1, 2.6])
    fig34.set_ylim([-2, 10])
    fig11.set_title(origin_traj_name + '_case_1')
    fig11.set_xlabel('X [m]')
    fig11.set_ylabel('Y [m]')
    fig12.set_title(origin_traj_name + '_case_2')
    fig12.set_xlabel('X [m]')
    fig12.set_ylabel('Y [m]')
    fig13.set_title(origin_traj_name + '_case_3')
    fig13.set_xlabel('X [m]')
    fig13.set_ylabel('Y [m]')
    fig14.set_title(origin_traj_name + '_case_4')
    fig14.set_xlabel('X [m]')
    fig14.set_ylabel('Y [m]')
    fig21.set_title(generate_traj_name + '_case_1')
    fig21.set_xlabel('X [m]')
    fig21.set_ylabel('Y [m]')
    fig22.set_title(generate_traj_name + '_case_2')
    fig22.set_xlabel('X [m]')
    fig22.set_ylabel('Y [m]')
    fig23.set_title(generate_traj_name + '_case_3')
    fig23.set_xlabel('X [m]')
    fig23.set_ylabel('Y [m]')
    fig24.set_title(generate_traj_name + '_case_4')
    fig24.set_xlabel('X [m]')
    fig24.set_ylabel('Y [m]')
    fig31.set_title(velocity_compare_name + '_case_1')
    fig31.set_xlabel('t [s]')
    fig31.set_ylabel('v [m/s]')
    fig32.set_title(velocity_compare_name + '_case_2')
    fig32.set_xlabel('t [s]')
    fig32.set_ylabel('v [m/s]')
    fig33.set_title(velocity_compare_name + '_case_3')
    fig33.set_xlabel('t [s]')
    fig33.set_ylabel('v [m/s]')
    fig34.set_title(velocity_compare_name + '_case_4')
    fig34.set_xlabel('t [s]')
    fig34.set_ylabel('v [m/s]')

    # origin_traj = origin_traj_lst[0]
    origin_traj = origin_traj_lst[0]
    recon_traj = generate_traj_lst[0]
    norm1 = plt.Normalize(origin_traj[:,0].min(), origin_traj[:,0].max())
    norm2 = plt.Normalize(recon_traj[:,0].min(), recon_traj[:,0].max())
    norm_lon1 = norm1(origin_traj[:,0])
    norm_lon2 = norm2(recon_traj[:,0])
    fig11.plot(origin_traj[:,0], origin_traj[:,1])
    fig11.scatter(origin_traj[:,0], origin_traj[:,1], c=norm_lon1, cmap='viridis')
    fig21.plot(recon_traj[:,0], recon_traj[:,1])
    fig21.scatter(recon_traj[:,0], recon_traj[:,1], c=norm_lon2, cmap='viridis')
    fig31.plot(origin_traj[:, 4], origin_traj[:, 3], 'bo')
    fig31.plot(recon_traj[:, 4], recon_traj[:, 3], 'r+')

    origin_traj = origin_traj_lst[1]
    recon_traj = generate_traj_lst[1]
    norm1 = plt.Normalize(origin_traj[:,0].min(), origin_traj[:,0].max())
    norm2 = plt.Normalize(recon_traj[:,0].min(), recon_traj[:,0].max())
    norm_lon1 = norm1(origin_traj[:,0])
    norm_lon2 = norm2(recon_traj[:,0])
    fig12.plot(origin_traj[:,0], origin_traj[:,1])
    fig12.scatter(origin_traj[:,0], origin_traj[:,1], c=norm_lon1, cmap='viridis')
    fig22.plot(recon_traj[:,0], recon_traj[:,1])
    fig22.scatter(recon_traj[:,0], recon_traj[:,1], c=norm_lon2, cmap='viridis')
    fig32.plot(origin_traj[:, 4], origin_traj[:, 3], 'bo')
    fig32.plot(recon_traj[:, 4], recon_traj[:, 3], 'r+')

    origin_traj = origin_traj_lst[2]
    recon_traj = generate_traj_lst[2]
    norm1 = plt.Normalize(origin_traj[:,0].min(), origin_traj[:,0].max())
    norm2 = plt.Normalize(recon_traj[:,0].min(), recon_traj[:,0].max())
    norm_lon1 = norm1(origin_traj[:,0])
    norm_lon2 = norm2(recon_traj[:,0])
    fig13.plot(origin_traj[:,0], origin_traj[:,1])
    fig13.scatter(origin_traj[:,0], origin_traj[:,1], c=norm_lon1, cmap='viridis')
    fig23.plot(recon_traj[:,0], recon_traj[:,1])
    fig23.scatter(recon_traj[:,0], recon_traj[:,1], c=norm_lon2, cmap='viridis')
    fig33.plot(origin_traj[:, 4], origin_traj[:, 3], 'bo')
    fig33.plot(recon_traj[:, 4], recon_traj[:, 3], 'r+')

    origin_traj = origin_traj_lst[3]
    recon_traj = generate_traj_lst[3]
    norm1 = plt.Normalize(origin_traj[:,0].min(), origin_traj[:,0].max())
    norm2 = plt.Normalize(recon_traj[:,0].min(), recon_traj[:,0].max())
    norm_lon1 = norm1(origin_traj[:,0])
    norm_lon2 = norm2(recon_traj[:,0])
    fig14.plot(origin_traj[:,0], origin_traj[:,1])
    fig14.scatter(origin_traj[:,0], origin_traj[:,1], c=norm_lon1, cmap='viridis')
    fig24.plot(recon_traj[:,0], recon_traj[:,1])
    fig24.scatter(recon_traj[:,0], recon_traj[:,1], c=norm_lon2, cmap='viridis')
    fig34.plot(origin_traj[:, 4], origin_traj[:, 3], 'bo')
    fig34.plot(recon_traj[:, 4], recon_traj[:, 3], 'ro')
    # plt.plot()
    plt.savefig(file_name)

    plot_img_np = get_img_from_fig(fig,dpi=180)
    plt.close(fig)
    return plot_img_np 

# def generate_compact_traj(traj, traj_len10, init_state, traj_len=10, mask=None):
#     total_traj_len = traj_len + 1 
#     traj = traj[:4, :, :]
#     init_state = init_state[:4, :]
#     batch_size = traj.shape[0]
#     traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1).to('cpu')
#     time = torch.arange(0,total_traj_len) / 10
#     time = time.squeeze(0).repeat(batch_size, 1)
#     traj = torch.cat([traj, time.unsqueeze(-1)], dim = 2)
    
#     return traj.detach().numpy()

# def generate_compact_traj(traj_len20, traj_len10, init_state, traj_len=10, trajj_type=None):
#     total_traj_len = traj_len + 1 
    
#     if trajj_type is not None:
#         uni_traj_list = []
#         for i in range(4):
#             uni_traj = torch.zeros_like(traj_len20[0])
#             uni_type = trajj_type[i]
#             if uni_type == 0:
#                 uni_traj[:10] = traj_len10[i] 
#                 fill_element = traj_len10[i,-1]
#                 fill_block = fill_element.repeat(10,1)
#                 uni_traj[10:] = fill_block 
#             else:
#                 uni_traj = traj_len20[i]
#             uni_traj_list.append(uni_traj)
#         traj = torch.stack(uni_traj_list, dim = 0)
#     else:
#         traj = traj_len20[:4, :, :]
    
#     #traj = traj[:4, :, :]
#     init_state = init_state[:4, :]
#     batch_size = traj.shape[0]
#     traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1).to('cpu')
#     time = torch.arange(0,total_traj_len) / 10
#     time = time.squeeze(0).repeat(batch_size, 1)
#     traj = torch.cat([traj, time.unsqueeze(-1)], dim = 2)
    
#     return traj.detach().numpy()

def generate_compact_traj(traj_len20, init_state, traj_len=10, trajj_type=None):
    total_traj_len = traj_len + 1 
    
    if trajj_type is not None:
        uni_traj_list = []
        for i in range(4):
            uni_traj = traj_len20[i]
            uni_traj_list.append(uni_traj)
        traj = torch.stack(uni_traj_list, dim = 0)
    else:
        traj = traj_len20[:4, :, :]
        traj = traj[:, :traj_len, :]
    
    #traj = traj[:4, :, :]
    init_state = init_state[:4, :]
    batch_size = traj.shape[0]
    init_state = init_state[:, :4]
    traj = traj[:, :, :4]
    traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1).to('cpu')
    time = torch.arange(0,total_traj_len) / 10
    time = time.squeeze(0).repeat(batch_size, 1)
    traj = torch.cat([traj, time.unsqueeze(-1)], dim = 2)
    
    return traj.detach().numpy()