import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys_path_utils 
from taec_utils.dir_utils import save_pickle, load_pickle, ensure_directory_exists, DATA_RAW_FOLDER
TRAJ_LIBRARY_PATH = DATA_RAW_FOLDER + '/traj/'

def load_pickle_files(folder_path):
    """
    从指定文件夹中加载所有 pickle 文件并返回解析后的数据。

    参数:
        folder_path (str): 包含 pickle 文件的文件夹路径。

    返回:
        list: 一个包含所有轨迹数据的列表，每个元素是一个 NumPy 数组。
    """
    trajs_list = []
    for filename in os.listdir(folder_path):
        # 检查是否是 pickle 文件
        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            file_path = os.path.join(folder_path, filename)
            trajs = load_pickle()
            trajs_list.append(trajs)
    return trajs_list


def visualize_trajectories(trajectories, output_path=None):
    """
    可视化所有轨迹。

    参数:
        trajectories (list): 包含所有轨迹数据的列表，每个元素是一个 NumPy 数组。
        output_path (str): 如果提供，则将图像保存到指定路径。
    """
    plt.figure(figsize=(10, 8))
    for i, trajectory in enumerate(trajectories):
        if trajectory.shape[1] < 2:
            print(f"Skipping trajectory {i+1}: insufficient dimensions.")
            continue
        x = trajectory[:, 0]  # x 坐标
        y = trajectory[:, 1]  # y 坐标
        plt.plot(x, y, label=f"Trajectory {i+1}")

    # 图形细节设置
    plt.title("Trajectories Visualization")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.axis("equal")  # 保持 x 和 y 轴比例一致

    # 保存或显示图像
    if output_path:
        plt.savefig(output_path)
        print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()


def main():
    # 用户输入文件夹路径
    folder_path = input("Please enter the folder path containing pickle files: ").strip()

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # 加载轨迹数据
    print("Loading pickle files...")
    trajectories = load_pickle_files(folder_path)

    if not trajectories:
        print("No valid pickle files found in the folder.")
        return

    # 可视化轨迹
    print("Visualizing trajectories...")
    visualize_trajectories(trajectories)


# 调用主函数
if __name__ == "__main__":
    main()