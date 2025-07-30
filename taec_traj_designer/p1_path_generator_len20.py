"""

State lattice planner with model predictive trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

- plookuptable.csv is generated with this script:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning
/ModelPredictiveTrajectoryGenerator/lookup_table_generator.py

Ref:

- State Space Sampling of Feasible Motions for High-Performance Mobile Robot
Navigation in Complex Environments
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.8210&rep=rep1
&type=pdf

"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import sys_path_utils 
from taec_utils.dir_utils import BASE_DIR
import taec_utils.ModelPredictiveTrajectoryGenerator.trajectory_generator as planner
import taec_utils.ModelPredictiveTrajectoryGenerator.motion_model as motion_model

TABLE_PATH = BASE_DIR + "/taec_utils/lookup_table.csv"
print(TABLE_PATH)
meta_x = 0.2 
meta_y = 0.1 
meta_theta = 1.0 
meta_vel = 0.2

scale_x = 1 / meta_x 
scale_y = 1 / meta_y 
scale_theta = 1 / meta_theta 

show_animation = True


pwd = os.getcwd()
DATA_PATH = pwd + '/data_folder/'
PATH_LIBRARY_PATH = pwd + '/dataset/metadrive/path_matfiles/'
print(PATH_LIBRARY_PATH)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(PATH_LIBRARY_PATH):
    os.makedirs(PATH_LIBRARY_PATH)
mode = 'draw' 
mode = 'save'


def search_nearest_one_from_lookup_table(t_x, t_y, t_yaw, lookup_table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):
        dx = t_x - table[0]
        dy = t_y - table[1]
        dyaw = t_yaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)
        if d <= mind:
            minid = i
            mind = d

    return lookup_table[minid]


def get_lookup_table(table_path):
    return np.loadtxt(table_path, delimiter=',', skiprows=1)


def generate_path(target_states, k0):
    # x, y, yaw, s, km, kf
    lookup_table = get_lookup_table(TABLE_PATH)
    result = []

    for state in target_states:
        bestp = search_nearest_one_from_lookup_table(
            state[0], state[1], state[2], lookup_table)

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [np.hypot(state[0], state[1]), bestp[4], bestp[5]]).reshape(3, 1)

        x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

        if x is not None:
            # print("find good path")
            result.append(
                [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])

    print("finish path generation")
    return result


def calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy):
    """

    calc lane states

    :param l_center: lane lateral position
    :param l_heading:  lane heading
    :param l_width:  lane width
    :param v_width: vehicle width
    :param d: longitudinal position
    :param nxy: sampling number
    :return: state list
    """
    xc = d
    yc = l_center

    states = []
    for i in range(nxy):
        delta = -0.5 * (l_width - v_width) + \
            (l_width - v_width) * i / (nxy - 1)
        xf = xc - delta * math.sin(l_heading)
        yf = yc + delta * math.cos(l_heading)
        yawf = l_heading
        states.append([xf, yf, yawf])

    return states



def lane_state_sampling_one_case(final_theta_degree, dd =20, show_animation = True):
    print('degree:')
    print(final_theta_degree)
    k0 = 0.0
    #l_center = 0.1 * final_theta_degree #2.4
    l_center = 0.04 * final_theta_degree #2.4
    # l_center = 0.04 * final_theta_degree #2.4
    l_heading = np.deg2rad(final_theta_degree)
    l_width = 15.0#5.0
    v_width = 1.0
    d = dd
    nxy = 25#15
    states = calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy)
    result = generate_path(states, k0)
    if show_animation:
        plt.close("all")
    path_list = []
    for table in result:
        x_c, y_c, yaw_c = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        x_new_c = []
        y_new_c = []
        theta_new_c = []
        # for i in range(8, len(x_c)):
        #     x_new_c.append(x_c[i-8] * 1.0)
        # for i in range(8, len(y_c)):
        #     y_new_c.append((y_c[i]-y_c[8])*1.0)
        #     theta_new_c.append(yaw_c[i])
        for i in range(15, len(x_c)):
            x_new_c.append(x_c[i-15] * 1.0)
        for i in range(15, len(y_c)):
            y_new_c.append((y_c[i]-y_c[15])*1.0)
            theta_new_c.append(yaw_c[i])
        for i in range(50):
            x_new_c.append(x_new_c[-1] + 0.1)
            y_new_c.append(y_new_c[-1] + 0.1 * np.tan(theta_new_c[-1]))
            theta_new_c.append(theta_new_c[-1])

        x_new_c = np.array(x_new_c)
        y_new_c = np.array(y_new_c)
        dist_threshold = 22
        refine_x_list = []
        refine_y_list = []
        refine_x_list.append(x_new_c[0])
        refine_y_list.append(y_new_c[0])
        
        accumulated_distance = 0
        for i in range(len(x_new_c)-1):
            distance = np.sqrt((x_new_c[i+1] - x_new_c[i])**2 + (y_new_c[i+1] - y_new_c[i])**2)
            accumulated_distance += distance 
            refine_x_list.append(x_new_c[i+1])
            refine_y_list.append(y_new_c[i+1])
            if accumulated_distance > dist_threshold:
                break 
        print(accumulated_distance)
        refine_theta_list = [0]
        for i in range(1, len(refine_x_list)):
            delta_x = refine_x_list[i] - refine_x_list[i-1]
            delta_y = refine_y_list[i] - refine_y_list[i-1]
            theta = np.arctan2(delta_y, delta_x)
            refine_theta_list.append(theta)

        refine_x = np.array(refine_x_list)
        refine_y = np.array(refine_y_list)
        refine_theta = np.array(refine_theta_list)
        lon = np.expand_dims(refine_x,1)
        lat = np.expand_dims(refine_y,1)
        yaw = np.expand_dims(refine_theta,1)
        yaw = yaw * 180 / 3.1415926
        path = np.hstack((lon, lat, yaw))
        path_list.append(path)
        if show_animation:
            plt.plot(refine_x, refine_y, "-r")
    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    return path_list

def main():
    import copy
    import scipy.io as io
    planner.show_animation = show_animation
    degree_list = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    degree_list = [-60, -50, -40, -30, -25 -20, -15, -10, -5, 0,5, 10, 15, 20, 25, 30, 40, 50, 60]
    # degree_list = [-40, -30, -25 -20, -15, -10, -5, 0,5, 10, 15, 20, 25, 30, 40]
    dd = 20 #20， 16， 12
    path_dictionary = {}
    
    for degree in degree_list:
        print('preparing degree {}'.format(degree))
        show_animation_zt = True 
        path_list = lane_state_sampling_one_case(degree, dd, show_animation_zt)
        end_theta_key = str(degree)
        need_refresh_theta = True
        for path in path_list:
            if path is not None:
                
                end_x = path[-1][0]
                end_y = path[-1][1]
                end_theta = path[-1][2]
                
                end_x_int = int(end_x * scale_x)
                end_y_int = int(end_y * scale_y)
                end_theta_int = int(end_theta * scale_theta)
                
                end_x_stand = float(end_x_int) *meta_x
                end_y_stand= float(end_y_int) * meta_y 
                end_theta_stand = float(end_theta_int) * meta_theta 
                
                end_x_stand = round(end_x_stand, 2)
                end_y_stand= round(end_y_stand, 2)
                end_theta_stand = round(end_theta_stand, 2)   
                
                if need_refresh_theta:
                    end_theta_key = str(end_theta_stand)
                    need_refresh_theta = False 
                    
                theta_diff = np.abs(float(end_theta_key) - end_theta)
                if theta_diff > 4:
                    continue
                end_x_key = str(end_x_stand)
                end_y_key = str(end_y_stand)
                
                if end_theta_key not in path_dictionary:
                    path_dictionary[end_theta_key] = {}
                path_dictionary[end_theta_key][end_y_key] = {}
                path_dictionary[end_theta_key][end_y_key]['path'] = path
                path_dictionary[end_theta_key][end_y_key]['x'] = end_x
                path_dictionary[end_theta_key][end_y_key]['y'] = end_y
                path_dictionary[end_theta_key][end_y_key]['theta'] = end_theta
                path_dictionary[end_theta_key][end_y_key]['xlabel'] = end_x_key
        print('zt1')
    print('zt2')
    import pickle 
    pkl_name = 'jan11_path_turn10m.pickle'
    pkl_name = PATH_LIBRARY_PATH + pkl_name 
    with open(pkl_name, "wb") as fp:
        pickle.dump(path_dictionary, fp)


if __name__ == '__main__':
    main()