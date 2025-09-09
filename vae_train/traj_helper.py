import torch
import os
import pdb
import glob
import math
import time
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s' % (asMinutes(s))

def print_loss(params, epoch, Training_loss, Validation_loss, Reconstruction_loss, KL_Div_loss, start_time):
    n_epochs = params.n_epochs
    if epoch % params.print_every_epoch == 0:
        print('epoch %s/%s(%d%%): %s (Training loss: %.4f) (Validation loss: %.4f) (Reconstruction loss: %.4f) (KL Divergence loss: %.4f)' % 
                (epoch, n_epochs, (epoch+1) / n_epochs * 100, timeSince(start_time, (epoch+1) / n_epochs),
                Training_loss[-1], Validation_loss[-1], Reconstruction_loss[-1], KL_Div_loss[-1]))

def save_model(epoch, params, model, best_epoch):
    # model_PATH = "result/{}/model/model_{}.pt".format(params.model_name, (epoch+params.restore_epoch))
    model_PATH = "result/{}/model/model_{}.pt".format(params.model_name, (epoch+params.restore_epoch))
    torch.save(model.state_dict(), model_PATH)

    # delete model before best epoch
    saved_model_dir = "result/model/model_*.pt"
    files = glob.glob(saved_model_dir)
    for file in files:
        pdb.set_trace()
        epoch_num = int(file.split('/')[-1].split('.')[6:])
        if epoch_num < best_epoch:
            os.remove(file)
