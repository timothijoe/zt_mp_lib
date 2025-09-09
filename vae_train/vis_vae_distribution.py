from torch.utils.data import Dataset, DataLoader
import torch
import os
import pdb
import glob
import numpy as np
import random
import copy
import scipy.io as io
import pickle
import matplotlib.pyplot as plt
''' data '''

SOURCE_TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/vis_folder/latent_traj_files/'
#TARGET_TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/target_picture_folder/'
TARGET_DISTRIBUTION_FOLDER = '/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/latent_distribution_folder/'

z_max = 1.0
z_min = -1.0
traj_file_list = []

for cur_file in os.listdir(SOURCE_TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(SOURCE_TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)


fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')

ax1.set_xlabel('z0')
ax1.set_ylabel('z1')
ax1.set_zlabel('z2')
ax1.set_xlim((-1.1,1.1))
ax1.set_ylim((-1.1,1.1))
ax1.set_zlim((-1.1,1.1))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('theta')
ax2.set_xlim((-1.1,21.1))
ax2.set_ylim((-10.1,10.1))
ax2.set_zlim((-45, 45))
zt =0
z0_list = []
z1_list = []
z2_list = []
x_list = []
y_list = []
theta_list = []
color_list = []

for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    # For each vel file, we load a traj_mat
    # firstly, we check the initial speed
    vel_name = os.path.basename(traj_file)
    vel_name = os.path.splitext(vel_name)[0]
    vel_str = vel_name.split("_")[1]
    vel = float(vel_str)
    print(vel)

    with open(traj_file, 'rb') as file:
        traj_mat = pickle.load(file)
    # For each vel file, we create a folder to store image
    (prezt, file_path_name) = os.path.split(traj_file)
    vel_folder_name = os.path.splitext(file_path_name)[0]
    vel_folder_name = TARGET_DISTRIBUTION_FOLDER + vel_folder_name
    if not os.path.exists(vel_folder_name):
        os.makedirs(vel_folder_name)

    for key, value in traj_mat.items():
        zt += 1
        latent_variable = value['latent_variable']
        trajectory = value['trajectory']
        # trajectory_len10 = value['trajectory_len10']
        z0 = latent_variable[0]
        z1 = latent_variable[1]
        z2 = latent_variable[2]
        #z2 = vel / 10.0
        #r_value = (z0-z_min)/(z_max-z_min)
        # g_value = (z1-z_min)/(z_max-z_min)
        b_value = 0.0
        r_value = 1.0

        b_value = (z2-z_min)/(z_max-z_min)
        r_value = (z1-z_min)/(z_max-z_min)
        #r_value = 1.0 if value['class_label'] == 0 else 0.0
        g_value = 1.0 #if value['class_label']== 1 else 0.0
        #b_value = 0.0 if value['class_label'] == 0 else 1.0
        z_color = (r_value, g_value, b_value)
        # x = trajectory[-1, 0] 
        # y = trajectory[-1, 1] 
        # theta = trajectory[-1,2]

        # x = trajectory_len10[-1, 0] 
        # y = trajectory_len10[-1, 1] 
        # theta = trajectory_len10[-1,2]

        x = trajectory[-1, 0] #if value['class_label']== 1 else trajectory_len10[-1,0]
        y = trajectory[-1, 1] #if value['class_label']== 1 else trajectory_len10[-1,1]
        theta = trajectory[-1,2] #if value['class_label']== 1 else trajectory_len10[-1,2]

        theta = theta * 180 / np.pi 
        z0_list.append(z0)
        z1_list.append(z1)
        z2_list.append(z2)
        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)
        color_list.append(z_color)
print('zt: {}'.format(zt))
print(len(x_list))
print(len(z0_list))
ax1.scatter3D(z0_list, z1_list, z2_list, s = 3.0, c=color_list, cmap='coolwarm')
ax2.scatter3D(x_list, y_list, theta_list, s = 3.0, c=color_list, cmap='coolwarm')
fig_name = vel_folder_name + '/latent_variable.pdf'
#plt.savefig(fig_name)
plt.show()
plt.close()

print('finished')

    

    


    
    