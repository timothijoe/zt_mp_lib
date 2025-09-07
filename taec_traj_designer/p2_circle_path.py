import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io
import copy

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER

PATH_LIBRARY_PATH = DATA_RAW_FOLDER + '/path/'
CIRCLE_LIBRARY_PATH = PATH_LIBRARY_PATH + '/circle_path/'
ensure_directory_exists(PATH_LIBRARY_PATH)
ensure_directory_exists(CIRCLE_LIBRARY_PATH)

print(CIRCLE_LIBRARY_PATH)
mode = 'draw' 
mode = 'save'

scaleMul = 0.75


def generate_from_control(v_0 = 1.0, w_0 = 0, duration = 30, dt = 0.1, save=False):
    duration_dt = int(duration / dt)
    x = 0
    y = 0
    theta =0
    v = v_0
    x_list = []
    y_list = []
    theta_list = []
    v_list = []
    for i in range(duration_dt):
        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)
        v_list.append(v)
        x = np.cos(theta) * v * dt + x
        y = np.sin(theta) * v * dt + y 
        theta = w_0 * dt + theta 
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    
    x_list = x_list * scaleMul
    y_list = y_list * scaleMul
    
    theta_list = np.array(theta_list)

    if save:
        x_list = np.expand_dims(x_list,1)
        y_list = np.expand_dims(y_list,1)
        theta_list = np.expand_dims(theta_list,1)
        return np.hstack((x_list, y_list, theta_list))
    else:
        plt.plot(x_list, y_list)


def generate_from_control_half(v_0 = 1.0, w_0 = 0, duration = 30, dt = 0.1, save=False):
    duration_dt = int(duration / dt)
    x = 0
    y = 0
    theta =0
    v = v_0
    x_list = []
    y_list = []
    theta_list = []
    v_list = []
    for i in range(duration_dt):
        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)
        v_list.append(v)
        x = np.cos(theta) * v * dt + x
        y = np.sin(theta) * v * dt + y 
        theta = w_0 * dt + theta 
        if theta > 3.14 / 1:
            break
        if theta < -3.14 / 1:
            break 
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    
    x_list = x_list * scaleMul
    y_list = y_list * scaleMul
    
    theta_list = np.array(theta_list)
    if save:
        x_list = np.expand_dims(x_list,1)
        y_list = np.expand_dims(y_list,1)
        theta_list = np.expand_dims(theta_list,1)
        return np.hstack((x_list, y_list, theta_list))
    else:
        plt.plot(x_list, y_list)

def plot_multi_control():
    v = 1.0
    for i in range(20, 40):
        if i % 3 != 0:
            if i > 32:
                continue
        r = i * 1.0
        w = v / r 
        generate_from_control(1.0, w)
    for i in range(20, 40):
        r = i * 0.5
        w = v / r 
        generate_from_control(1.0, w)
    for i in range(7, 20):
        r = i * 0.5
        w = v / r                     
        generate_from_control_half(1.0, w)

    for i in range(20, 40):
        if i % 3 != 0:
            if i > 32:
                continue
        r = -i * 1.0
        w = v / r 
        generate_from_control(1.0, w)
    for i in range(20, 40):
        r = -i * 0.5
        w = v / r 
        generate_from_control(1.0, w)
    for i in range(7, 20):
        r = -i * 0.5
        w = v / r                     
        generate_from_control_half(1.0, w)
    
    for i in range(-24, 25, 3):
        w = 0.001 * i
        generate_from_control_half(1.0, w)
        
    # for i in range(0, 300, 3):
    #     w = 0.001 * i
    #     generate_from_control_half(1.0, w)
        
    plt.show()


def save_multi_control():
    path_id = 0
    path_dictionary = {}
    v = 1.0
    for i in range(20, 40):
        if i % 3 != 0:
            if i > 32:
                continue
        r = i * 1.0
        w = v / r 
        path = generate_from_control(1.0, w, save=True)
        path_dictionary[str(path_id)] = copy.deepcopy(path)
        path_id += 1
    for i in range(20, 40):
        r = i * 0.5
        w = v / r 
        path = generate_from_control(1.0, w, save=True)
        path_dictionary[str(path_id)] = copy.deepcopy(path)
        path_id += 1
    for i in range(7, 20):
        r = i * 0.5
        w = v / r                     
        path = generate_from_control_half(1.0, w, save=True)
        path_dictionary[str(path_id)] = copy.deepcopy(path)
        path_id += 1
    for i in range(20, 40):
        if i % 3 != 0:
            if i > 32:
                continue
        r = -i * 1.0
        w = v / r 
        path = generate_from_control(1.0, w, save=True)
    for i in range(20, 40):
        r = -i * 0.5
        w = v / r 
        path = generate_from_control(1.0, w, save=True)
        path_dictionary[str(path_id)] = copy.deepcopy(path)
        path_id += 1
    for i in range(7, 20):
        r = -i * 0.5
        w = v / r                     
        path = generate_from_control_half(1.0, w, save=True)
        path_dictionary[str(path_id)] = copy.deepcopy(path)
        path_id += 1
    for i in range(-24, 25, 3):
        w = 0.001 * i
        path = generate_from_control_half(1.0, w, save=True)
        path_dictionary[str(path_id)] = copy.deepcopy(path)
        path_id += 1
        
    mat_name = CIRCLE_LIBRARY_PATH + "control_primitive_" + str(path_id) + ".pkl" 
    save_pickle(mat_name, path_dictionary) 

if mode=='draw':
    # generate_from_control()
    plot_multi_control()

if mode=='save':
    save_multi_control()