import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io
import copy
import random


import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER

# PATH_LIBRARY_PATH = father_path + '/dataset/raw_data/path_matfiles'   #remove the last '/'
# VEL_LIBRARY_PATH = father_path + '/dataset/raw_data/vel_matfiles'     #remove the last '/'


PATH_LIBRARY_PATH = DATA_RAW_FOLDER + '/path/lattice/'   #remove the last '/'
# PATH_LIBRARY_PATH = DATA_RAW_FOLDER + '/path/circle_path/'   #remove the last '/'
# PATH_LIBRARY_PATH = DATA_RAW_FOLDER + '/path/straight_path/'   #remove the last '/'
VEL_LIBRARY_PATH = DATA_RAW_FOLDER + '/vel/'
TRAJ_LIBRARY_PATH = DATA_RAW_FOLDER + '/traj/'
ensure_directory_exists(TRAJ_LIBRARY_PATH)

class PathClusterParam():
    def __init__(self, path_mat_file):
        self.path_cluster_dict = load_pickle(path_mat_file)
        self.path_num = len(self.path_cluster_dict.keys())
        self.path_id_list = list(self.path_cluster_dict.keys())
    def select_rdm_path(self):
        id = random.choice(self.path_id_list)
        random_path = MetaPath(self.path_cluster_dict[id])        
        #random_path.PlotPath()
        return random_path

class VelClusterParam():
    def __init__(self, vel_mat_file):
        self.vel_cluster_dict = load_pickle(vel_mat_file)
        self.vel_num=len(self.vel_cluster_dict.keys())
        self.vel_id_list = list(self.vel_cluster_dict.keys())
    def select_rdm_vel(self):
        id = random.choice(self.vel_id_list)
        random_vel = MetaVel(self.vel_cluster_dict[id])
        #random_vel.PlotSpeed()
        return random_vel
        
class MetaPath():
    def __init__(self, wp_list):
        self.path = wp_list 
        self.path_length = self.getLen()
        self.total_path_len = self.path_length[-1]

    def GetPosFromLength(self, s):
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))
        if(self.path_length[-1] - s < 0):
            print(self.path_length[-1])
            print('zt')
            print(s)
        matched_ind = min(np.where(np.array(self.path_length)-s >= 0)[0])
        if matched_ind == 0:
            return self.path[:1,:]
        else:
            pos = np.zeros((1,3))         
            percent = (s - cal_length(self.path[:matched_ind, :])) / cal_length(self.path[matched_ind-1:matched_ind+1, :])
            pos[0,0] = self.path[matched_ind-1, 0] + percent * (self.path[matched_ind, 0] - self.path[matched_ind-1, 0])
            pos[0,1] = self.path[matched_ind-1, 1] + percent * (self.path[matched_ind, 1] - self.path[matched_ind-1, 1])
            pos[0,2] = self.path[matched_ind-1, 2] + percent * (self.path[matched_ind, 2] - self.path[matched_ind-1, 2])
            return pos
    def getLen(self):
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))
        path_length = [cal_length(self.path[:i+1,:2]) for i in range(len(self.path-1))]
        return path_length            
    def PlotPath(self):
        plt.title('Path Profile')
        plt.plot(self.path[:,0], self.path[:,1])

class MetaVel():
    def __init__(self, vel_list):
        self.vel_list = vel_list 
        self.t = self.vel_list[0,:]
        self.s = self.vel_list[1,:]
        self.milestone = [0]
        self.total_dist = self.getIntegral()
        self.T = self.t[-1]

    def getIntegral(self):
        self.dt = self.t[1] - self.t[0]
        dist = 0
        for i in range(len(self.s)):
            if i == len(self.s) - 1:
                continue  
            speed = self.s[i]
            dist += speed * self.dt
            self.milestone.append(dist)
        return dist

    def get_speed_list(self):
        return self.s
    def GetDistance(self, t):
        if t > self.T:
            raise Exception ('GetSpeed method: The specified time exceeds the time horizon of speed profile')
        else:
            return self.milestone[int(round(t/self.dt))]
    def GetSpeed(self, t):
        if t > self.T:
            raise Exception ('GetSpeed method: The specified time exceeds the time horizon of speed profile')
        else:
            return self.s[int(round(t/self.dt))]
    def PlotSpeed(self):
        plt.title('Speed Profile')
        plt.plot(self.t, self.s)

path_file_list = []
vel_file_list = []

for cur_file in os.listdir(PATH_LIBRARY_PATH):
    cur_path = os.path.join(PATH_LIBRARY_PATH, cur_file)
    path_file_list.append(cur_path)
for cur_file in os.listdir(VEL_LIBRARY_PATH):
    cur_path = os.path.join(VEL_LIBRARY_PATH, cur_file)
    vel_file_list.append(cur_path)

print(path_file_list)
print(vel_file_list)
mode = 'save'
# mode = 'draw'
if mode == 'save':
    for path_file in path_file_list:
        print('Preparing for reading path file: {}'.format(path_file))
        (prezt, file_path_name) = os.path.split(path_file)

        #path_elem_name = file_path_name.strip('degree')
        path_elem_name = file_path_name.strip('.pkl')
        path_elem_name = 'path_' + path_elem_name
        path_cluster = PathClusterParam(path_file)
        for vel_file in vel_file_list:
            print('Preparing for reading vel: {}'.format(vel_file))
            (prezt, file_vel_name) = os.path.split(vel_file)
            vel_elem_name = file_vel_name.strip('vel_init_')
            vel_elem_name = vel_elem_name.strip('.pkl')
            vel_elem_name = 'vel_' + vel_elem_name
            vel_cluster = VelClusterParam(vel_file)
            traj_dict = {}
            ind = 0
            for path_id in path_cluster.path_id_list:
                path = MetaPath(path_cluster.path_cluster_dict[path_id])
                for vel_id in vel_cluster.vel_id_list:
                    vel = MetaVel(vel_cluster.vel_cluster_dict[vel_id])
                    if(vel.total_dist > path.total_path_len):
                        print('zt')
                        print(vel.total_dist)
                        print(path.total_path_len)
                        continue 
                    else:
                        Ros = np.zeros((1,5))
                        SpeedCoveredDist = []
                        Ros[0][3] = vel.s[0]
                        for t in vel.t:
                            if t == 0.0:
                                continue
                            SpeedCoveredDist.append(vel.GetDistance(t))
                            cur_vel = vel.GetSpeed(t)
                            cur_vel = np.array([[cur_vel]])
                            cur_time = np.array([[t]])
                            pose_twist = np.hstack((path.GetPosFromLength(SpeedCoveredDist[-1]),cur_vel, cur_time))
                            Ros = np.vstack((Ros, pose_twist))
                        traj_dict[str(ind)] = Ros
                        ind += 1
            mat_name = TRAJ_LIBRARY_PATH + 'traj' + path_elem_name + '_'+vel_elem_name+'.pkl'
            save_pickle(mat_name, traj_dict) 
            #print(mat_name)

if mode == 'draw':
    random_path_file = random.choice(path_file_list)
    print(random_path_file)
    random_vel_file = random.choice(vel_file_list)
    path_cluster = PathClusterParam(random_path_file)
    vel_cluster = VelClusterParam(random_vel_file)
    path = path_cluster.select_rdm_path()
    vel = vel_cluster.select_rdm_vel()
    while(vel.total_dist > path.total_path_len):
        print('zt')
        print(vel.total_dist)
        print(path.total_path_len)
        path = path_cluster.select_rdm_path()
        vel = vel_cluster.select_rdm_vel()
    Ros = np.zeros((1,5))
    SpeedCoveredDist = []
    Ros[0][3] = vel.s[0]
    for t in vel.t:
        if t == 0.0:
            continue
        SpeedCoveredDist.append(vel.GetDistance(t))
        cur_vel = vel.GetSpeed(t)
        cur_vel = np.array([[cur_vel]])
        cur_time = np.array([[t]])
        pose_twist = np.hstack((path.GetPosFromLength(SpeedCoveredDist[-1]),cur_vel, cur_time))
        Ros = np.vstack((Ros, pose_twist))
    norm = plt.Normalize(Ros[:,0].min(), Ros[:,0].max())
    norm_lon = norm(Ros[:,0])
    print(Ros[:,:4])
    plt.scatter(Ros[:,0], Ros[:,1], c=norm_lon, cmap='viridis')
    plt.xlim(0, 15)  # 设置 x 轴范围
    plt.ylim(-8, 8)  # 设置 y 轴范围
    
    plt.show()