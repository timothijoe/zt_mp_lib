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
from vis_helper import save_traj_to_img, generate_compact_traj, save_trajs_to_img_batch
import os 
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_model(model):
    # model_PATH = "result/{}/model/model_{}.pt".format(params.model_name, (epoch+params.restore_epoch))
    model_PATH = "zt_model1.pt"
    torch.save(model.state_dict(), model_PATH)

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
    parser.add_argument('--n_epochs', type=int, default=150) # 100
    args = parser.parse_args()
    params = hyper_parameter(args)
    mk_logdir(params)
    train_dataset, validation_dataset, train_loader, validation_loader = load_train_eval(params,test = params.test)
    ''' create model '''
    model = create_model(params)

    vae_model_dir = '/home/zhoutong/dec_jan/traj_data_process/result/zt_jan13_005/ckpt/40_ckpt'
    vae_model_dir = None 
    if vae_model_dir is not None:
        checkpoint = torch.load(vae_model_dir)
        model.load_state_dict(checkpoint)

    # ''' restore model '''
    # if params.restore_model:
    #     model, restored_epoch = restore_model(model, params, epoch=params.restore_epoch)
    # checkpoint = torch.load('/home/SENSETIME/zhoutong/hoffnung/motion_primitive_vae/result/Feb02/ckpt/85_ckpt')

    # model.load_state_dict(checkpoint)
    ''' train model '''
    if params.train_the_model:
        #Min_val_loss_epoch = train_net(model, full_loader, params)
        Min_val_loss_epoch = train_vae(model, train_loader, validation_loader, params)

def train_vae(model, train_loader, val_loader, params):
    start_time = time.time()
    exp_name = params.exp_name
    tb_logger = SummaryWriter('result/{}/log/'.format(exp_name))
    #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay = params.adam_weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    noise_scale = 0.1
    iter_num = 0
    current_epoch = -1
    for epoch in range(params.n_epochs + 1):
        train_epoch_item = {}
        model.train()
        decs = 'Train - epoch-{}'.format(epoch)
        #for train_init, train_traj in tqdm(train_loader, desc = 'Train'):
        for train_init, train_traj, traj_type, traj_mask_0, traj_mask in tqdm(train_loader, desc = decs):
            train_init = train_init.float().to(params.device)
            train_traj = train_traj.float().to(params.device)
            traj_mask = traj_mask.float().to(params.device)
            traj_mask_0 = traj_mask_0.float().to(params.device)
            traj_type = traj_type.float().to(params.device)
            ret = model.forward(train_traj, train_init, traj_type, noise_scale)

            recons = generate_compact_traj(ret[0], train_init, params.seq_len, trajj_type = traj_type)
            expers = generate_compact_traj(ret[1], train_init, params.seq_len, trajj_type = None)
            name = ['train_expert', 'reconstruct', exp_name, str(epoch)]

            ret = model.loss_function(*ret, traj_mask, traj_mask_0, epoch)
            loss = ret['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in ret.items():
                v_record = v.item() if isinstance(v, torch.Tensor) else v 
                tb_logger.add_scalar("A_train_iter/{}".format(k), v_record, iter_num)
                if k not in train_epoch_item.keys():
                    train_epoch_item[k] = v_record 
                else:
                    train_epoch_item[k] += v_record 
            iter_num += 1   
        for k, v in train_epoch_item.items():
            v_record = v / len(train_loader)
            tb_logger.add_scalar("B_train_epoch/{}".format(k), v_record, epoch)
            
        if (epoch > 50 and epoch < 100):
            scheduler.step()
            noise_scale = min(0.1 - epoch / 1250, 0.1)
            noise_scale = max(0.001, noise_scale)
        
        if epoch % params.val_freq != 0:
            continue 
        # starting evaluate
        model.eval()
        eval_batch_index = 0
        eval_epoch_item = {}
        for val_init, val_traj, traj_type, traj_mask_0, traj_mask in tqdm(val_loader, desc = 'Val  '):
            with torch.no_grad():
                val_init = val_init.float().to(params.device)
                val_traj = val_traj.float().to(params.device)
                traj_mask = traj_mask.float().to(params.device)
                traj_mask_0 = traj_mask_0.float().to(params.device)
                traj_type = traj_type.float().to(params.device)
                ret = model.forward(val_traj, val_init, traj_type, noise_scale)
                
                recons = generate_compact_traj(ret[0], train_init, params.seq_len, trajj_type = traj_type)
                expers = generate_compact_traj(ret[1], train_init, params.seq_len, trajj_type = None)
                ret = model.loss_function(*ret, traj_mask, traj_mask_0, epoch)
                name = ['train_expert', 'reconstruct', exp_name, str(epoch), str(eval_batch_index)]
                for k, v in ret.items():
                    v_record = v.item() if isinstance(v, torch.Tensor) else v 
                    tb_logger.add_scalar("C_eval_iter/{}".format(k), v_record, iter_num)
                    if k not in eval_epoch_item.keys():
                        eval_epoch_item[k] = v_record 
                    else:
                        eval_epoch_item[k] += v_record 
                eval_batch_index += 1 
                if epoch % params.val_save_fig == 0 and epoch > 0:
                    save_trajs_to_img_batch(name, expers, recons)
        for k, v in eval_epoch_item.items():
            v_record = v / len(val_loader)
            tb_logger.add_scalar("D_eval_epoch/{}".format(k), v_record, epoch)
                       
        if (epoch % params.val_save_ckpt != 0) or (epoch == 0):
            continue 
        state_dict = model.state_dict()
        torch.save(state_dict, "result/{}/ckpt/{}_ckpt".format(exp_name, epoch))   
        torch.save(model.vae_decoder.state_dict(), "result/{}/ckpt/{}_decoder_ckpt".format(exp_name, epoch))  
        torch.save(model.vae_encoder.state_dict(), "result/{}/ckpt/{}_encoder_ckpt".format(exp_name, epoch))    







if __name__ == '__main__':
    main()
