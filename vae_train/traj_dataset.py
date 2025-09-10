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

''' data '''
# params.track_lst = '/home/SENSETIME/zhoutong/hoffnung/motion_primitive_library/data/traj_matfiles'
pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep+".")
TRAJ_LIBRARY_PATH_TRAIN = pwd + '/dataset/train_traj_data'   #remove the last '/'
TRAJ_LIBRARY_PATH_EVAL = pwd + '/dataset/eval_traj_data'   #remove the last '/'

traj_file_list = []
eval_file_list = []
for cur_file in os.listdir(TRAJ_LIBRARY_PATH_TRAIN):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH_TRAIN, cur_file)
    traj_file_list.append(cur_traj)
for cur_file in os.listdir(TRAJ_LIBRARY_PATH_EVAL):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH_EVAL, cur_file)
    eval_file_list.append(cur_traj)

class trajDataset(Dataset):
    def __init__(self, data_mode = 'train', test = False):
        self.test = test 
        self.data_mode = data_mode 
        self.split_ratio = 0.99
        self.extraced_data = self.read_data()
        self.len = len(self.extraced_data)

    def read_data(self):
        self.all_training_files = []
        self.all_validation_files = []
        traj_library = {}
        if self.test == False:
            data_file_list = traj_file_list 
        else: 
            data_file_list = eval_file_list
            data_file_list = traj_file_list 

        for traj_file in data_file_list:
            with open(traj_file, 'rb') as file:
                traj_mat = pickle.load(file)
            for key, value in traj_mat.items():
                # value is a dictionary that has label and trajectory
                library_key = len(traj_library)
                traj_library[str(library_key)] = {}
                traj_library[str(library_key)]['traj_type'] = value['traj_type']
                traj_library[str(library_key)]['traj_mask_0'] = value['traj_weight'].transpose(1,0)
                traj_library[str(library_key)]['traj_mask_1'] = value['traj_weight'].transpose(1,0)
                traj_library[str(library_key)]['raw_trajectory'] = value['raw_trajectory']
                traj_library[str(library_key)]['track_trajectory'] = value['track_trajectory']
                traj_library[str(library_key)]['track_action'] = value['track_action']
                traj_library[str(library_key)]['trajectory'] = np.concatenate((value['track_trajectory'], value['track_action']), axis = 1)
                # 0-4: x, y, theta, v, t,   5-6: acc, steer
                # traj_library[str(library_key)] = value['trajectory'].transpose(1, 0)

        # mode 'full' returns all trajectories
        if self.data_mode == 'full':
            return traj_library 
        key_list = [i for i in traj_library.keys()]

        training_keys = random.sample(key_list, int(len(key_list)* self.split_ratio))
        if self.data_mode == 'train':
            train_library = {}
            for key in training_keys:
                train_key = len(train_library)
                train_library[str(train_key)] = traj_library[key]
            return train_library
        else:
            validation_library_raw = copy.deepcopy(traj_library)
            validation_library = {}
            for key in training_keys:
                validation_library_raw.pop(key)
            for key in validation_library_raw.keys():
                validation_key = len(validation_library)
                validation_library[str(validation_key)] = validation_library_raw[key]            
            return validation_library


    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        # self.extraced_data.shape = 5, 11,
        # x, y, theta, v, time
        return self.extraced_data[str(idx)]['trajectory'][0,[0,1,2,3,5,6]], self.extraced_data[str(idx)]['trajectory'][1:,[0,1,2,3,5,6]], \
            self.extraced_data[str(idx)]['traj_type'],self.extraced_data[str(idx)]['traj_mask_0'][:,[0,1,2,3]], \
            self.extraced_data[str(idx)]['traj_mask_1'][:,[0,1,2,3]]


def load_data(params):
    full_dataset = trajDataset(params, data_mode = 'full')
    # train_dataset = trajDataset(params, data_mode = 'train')
    # val_dataset = trajDataset(params, data_mode = 'val')
    full_loader = torch.utils.data.DataLoader(dataset= full_dataset, batch_size = params.batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)
    return full_dataset, full_loader

def load_train_eval(params, test = True):
    train_dataset= trajDataset(data_mode = 'full', test=test)
    validation_dataset = trajDataset(data_mode = 'validation', test=test)
    train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = params.batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(dataset= validation_dataset, batch_size = params.batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)
    return train_dataset, validation_dataset, train_loader, validation_loader