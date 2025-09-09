# from traj_model import create_model, VAE_loss
from traj_vae import create_model
from parameter import hyper_parameter
import numpy as np
import time, copy
import pdb
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
    mk_logdir(params)
    train_dataset, validation_dataset, train_loader, validation_loader = load_train_eval(params,test = params.test)
    ''' create model '''
    model = create_model(params)

    pwd = os.getcwd()
    checkpoint = torch.load('/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/zNov-v4-len10-dim3-notanh/ckpt/99_ckpt')
    model.load_state_dict(checkpoint)

    a = np.array([])
    device = torch.device('cuda')
    LT_PATH = '/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/vis_folder/latent_traj_files/'
    for v in range(0, 10): #10
        name = 'vel_{}.pickle'.format(v)
        traj_dict = {}
        v = v * 1.0
        init_state = np.array([0, 0, 0, v])
        init_state = torch.from_numpy(init_state)
        traj_list = []
        latent_list = []
        # c means count, v means value
        for c_z0 in range(0, 21):
            v_z0 = c_z0 * 0.1 - 1.0
            z_list = []
            init_state_list = []
            for c_z1 in range(0, 21):
                v_z1 = c_z1 * 0.1 - 1.0

                # latent_v = np.array([[v_z0, v_z1]])
                # latent_v = torch.from_numpy(latent_v)
                # z_list.append(latent_v)
                # init_state_cp = copy.deepcopy(init_state)
                # init_state_list.append(init_state_cp)

                for c_z2 in range(0, 21):
                    v_z2 = c_z2 * 0.1 - 1.0

                    latent_v = np.array([[v_z0, v_z1, v_z2 ]])
                    latent_v = torch.from_numpy(latent_v)
                    z_list.append(latent_v)
                    init_state_cp = copy.deepcopy(init_state)
                    init_state_list.append(init_state_cp)

                    # for c_z3 in range(0, 21):
                    #     v_z3 = c_z3 * 0.1 - 1.0

                    #     latent_v = np.array([[v_z0, v_z1, v_z2, v_z3]])
                    #     latent_v = torch.from_numpy(latent_v)
                    #     z_list.append(latent_v)
                    #     init_state_cp = copy.deepcopy(init_state)
                    #     init_state_list.append(init_state_cp)

                        # for c_z4 in range(0, 21):
                        #     v_z4 = c_z4 * 0.1 - 1.0

                        #     latent_v = np.array([[v_z0, v_z1, v_z2, v_z3, v_z4 ]])
                        #     latent_v = torch.from_numpy(latent_v)
                        #     z_list.append(latent_v)
                        #     init_state_cp = copy.deepcopy(init_state)
                        #     init_state_list.append(init_state_cp)


            z_batch = torch.stack(z_list, dim = 1).to(torch.float32).to(device)
            init_state_batch = torch.stack(init_state_list, dim=0).to(torch.float32).to(device)
            construct_trajs = model.sample(z_batch, init_state_batch)
            construct_trajs = torch.cat([init_state_batch.unsqueeze(1), construct_trajs], dim = 1)
            # construct_trajs_len10 = torch.cat([init_state_batch.unsqueeze(1), construct_trajs_len10], dim = 1)
            traj_num = len(construct_trajs)
            for i in range(0,traj_num):
                traj = construct_trajs[i]
                #traj = construct_trajs_len10[i]
                # traj_len10 = construct_trajs_len10[i]
                latent_variable = z_batch[0][i]
                init_state = init_state_batch[i]
                dict_key = str(len(traj_dict))
                # class_label = torch.argmax(output_labels[0][i])
                # class_label = class_label.data.cpu().numpy()
                single_traj = {}
                single_traj['latent_variable'] = latent_variable.data.cpu().numpy() 
                single_traj['trajectory'] = traj.data.cpu().numpy() 
                # single_traj['trajectory_len10'] = traj_len10.data.cpu().numpy() 
                single_traj['init_vel'] = v
                # single_traj['class_label'] = class_label
                #single_traj['init_state'] = init_state.data.cpu().numpy() 
                traj_dict[dict_key] = single_traj
        print('init vel : {}'.format(v))
        pkl_name = LT_PATH + name 
        with open(pkl_name, "wb") as fp:
            pickle.dump(traj_dict, fp)


if __name__ == '__main__':
    main()
