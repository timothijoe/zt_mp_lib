import numpy as np
import matplotlib.pyplot as plt
import copy

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER

PATH_LIBRARY_PATH = DATA_RAW_FOLDER + '/path/'
STRAIGHT_LIBRARY_PATH = PATH_LIBRARY_PATH + '/straight_path/'
ensure_directory_exists(PATH_LIBRARY_PATH)
ensure_directory_exists(STRAIGHT_LIBRARY_PATH)

print(STRAIGHT_LIBRARY_PATH)

def generate_straight_line_trajectory(duration, dt, v_0, scaleMul=1.0, save=True):
    """
    生成直线轨迹并存储其位置信息 (x, y) 和方向信息 (theta)。
    
    参数:
        duration (float): 模拟的总时间 (秒)。
        dt (float): 时间步长 (秒)。
        v_0 (float): 初始速度 (单位: 米/秒)。
        scaleMul (float): 缩放比例因子，用于缩放 x 和 y 坐标。
        save (bool): 如果为 True，则返回存储的轨迹数据。

    返回:
        np.ndarray: 包含轨迹数据的数组，格式为 [[x, y, theta], ...]。
    """
    # 计算时间步数
    duration_dt = int(duration / dt)
    
    # 初始化状态变量
    x, y, theta = 0, 0, 0  # 初始位置 (x, y) 和方向 theta
    v = v_0  # 初始速度

    # 用于存储轨迹点的列表
    x_list, y_list, theta_list = [], [], []

    # 模拟直线轨迹
    for _ in range(duration_dt):
        # 存储当前状态
        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)

        # 更新状态（直线运动，角度 theta 不变）
        x += np.cos(theta) * v * dt
        y += np.sin(theta) * v * dt

    # 转为 NumPy 数组并缩放
    x_array = np.array(x_list) * scaleMul
    y_array = np.array(y_list) * scaleMul
    theta_array = np.array(theta_list)

    # 如果需要保存，返回轨迹数据
    x_array = np.expand_dims(x_array, 1)  # 转为列向量
    y_array = np.expand_dims(y_array, 1)  # 转为列向量
    theta_array = np.expand_dims(theta_array, 1)  # 转为列向量
    return np.hstack((x_array, y_array, theta_array))  # 合并为二维数组


def main():
    # 参数设置
    duration = 10  # 模拟总时间 10 秒
    dt = 0.1       # 时间步长 0.1 秒
    v_0 = 1.0      # 初始速度 1 米/秒
    scaleMul = 1.0 # 缩放比例因子
    mode = 'draw' 
    mode = 'save'
    # 生成直线轨迹
    trajectory = generate_straight_line_trajectory(duration, dt, v_0, scaleMul, save=True)

    # 打印轨迹数据
    print("Trajectory Data (x, y, theta):")
    print(trajectory)
    if mode == 'draw':
        # 提取 x 和 y 坐标
        x_vals = trajectory[:, 0]
        y_vals = trajectory[:, 1]

        # 可视化轨迹
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="Straight Line Trajectory", color="blue")
        plt.scatter(x_vals[0], y_vals[0], color="red", label="Start Point")  # 起点
        plt.scatter(x_vals[-1], y_vals[-1], color="green", label="End Point")  # 终点
        plt.title("Generated Straight Line Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.show()
    if mode == 'save':
        path_dictionary = {}
        path_id = 0
        path_dictionary[str(path_id)] = copy.deepcopy(trajectory)
        mat_name = STRAIGHT_LIBRARY_PATH + "straight_line" + str(path_id) + ".pkl" 
        save_pickle(mat_name, path_dictionary) 

# 调用主函数
if __name__ == "__main__":
    main()