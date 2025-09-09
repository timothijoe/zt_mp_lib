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
    parser.add_argument('--n_epochs', type=int, default=80)
    args = parser.parse_args()
    params = hyper_parameter(args)
    mk_logdir(params)
    train_dataset, validation_dataset, train_loader, validation_loader = load_train_eval(params,test = params.test)
    ''' create model '''
    model = create_model(params)

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
    iter_num = 0
    current_epoch = -1

    for epoch in range(params.n_epochs):
        model.train()
        decs = 'Train - epoch-{}'.format(epoch)
        #for train_init, train_traj in tqdm(train_loader, desc = 'Train'):
        for train_init, train_traj, train_label, train_mask in tqdm(train_loader, desc = decs):
            train_init = train_init.float().to(params.device)
            train_traj = train_traj.float().to(params.device)
            ret = model.forward(train_traj, train_init)

            recons = generate_compact_traj(ret[0], train_init)
            expers = generate_compact_traj(ret[1], train_init)
            name = ['train_expert', 'reconstruct', exp_name, str(epoch)]
            fake_epoch = iter_num // 50
            
            ret = model.loss_function(*ret, epoch)
            loss = ret['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_num % 50 == 0:
                for k, v in ret.items():
                    tb_logger.add_scalar("train_iter/{}".format(k), v.item(), iter_num)

            if epoch != current_epoch:
                current_epoch = epoch
                for k, v in ret.items():
                    tb_logger.add_scalar("train_epoch/{}".format(k), v.item(), epoch)
                img_to_tensorboard = save_traj_to_img(name, expers, recons)
                tb_logger.add_image('train_epoch/train_recon_comprare', img_to_tensorboard, epoch, dataformats='HWC')
            iter_num += 1
        if epoch > 0 and epoch % params.val_freq == 0:
            model.eval()
            total_loss = {}
            for val_init, val_traj in tqdm(val_loader, desc = 'Val  '):
                with torch.no_grad():
                    val_init = val_init.float().to(params.device)
                    val_traj = val_traj.float().to(params.device)
                    ret = model.forward(val_traj, val_init)
                    loss = model.loss_function(*ret)
                    for k, v in loss.items():
                        if k not in total_loss:
                            total_loss[k] = [v]
                        else:
                            total_loss[k].append(v)
            total_loss_mean = {k: torch.stack(v).mean().item() for k, v in total_loss.items()}
            for k, v in total_loss_mean.items():
                tb_logger.add_scalar("val_epoch/{}_avg".format(k), v, epoch)
            
            # test_init, test_sample = next(iter(val_loader))
            # test_init, test_sample = test_init.float().to(params.device), test_sample.float().to(params.device)
            # test_init = test_init[:16].to(params.device)
            # test_sample = test_sample[:16].to(params.device)
            # with torch.no_grad():
            #     recon_sample = model.generate(test_sample, test_init)
            #     random_sample = model.sample(16, test_init)

            # tb_logger.add_image('rec_bev', (recon_sample), epoch)
            # tb_logger.add_image('ran_bev', (random_sample), epoch)  
            # if not os.path.exists('./ckpt'):
            #     os.makedirs('.ckpt')
            state_dict = model.state_dict()
            torch.save(state_dict, "result/{}/ckpt/{}_ckpt".format(exp_name, epoch))   
            torch.save(model.vae_decoder.state_dict(), "result/{}/ckpt/{}_decoder_ckpt".format(exp_name, epoch))  
            torch.save(model.vae_encoder.state_dict(), "result/{}/ckpt/{}_encoder_ckpt".format(exp_name, epoch))    







if __name__ == '__main__':
    main()
