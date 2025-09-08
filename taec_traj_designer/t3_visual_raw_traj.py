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
        dict: 包含文件名和解析后的轨迹数据，每个文件的数据是一个列表。
        格式: {filename: [trajectory1, trajectory2, ...]}
    """
    trajectories_dict = {}
    for filename in os.listdir(folder_path):
        # 检查是否是 pickle 文件
        if filename.endswith(".pkl") or filename.endswith(".pickle"):
            file_path = os.path.join(folder_path, filename)
            data = load_pickle(file_path)
            trajectories_dict[filename] = [np.array(trajectory) for key, trajectory in data.items()]
    return trajectories_dict


def plot_trajectories(trajectories, output_folder, filename):
    """
    绘制一个文件中的多个轨迹，并保存为单独的图像。

    参数:
        trajectories (list): 包含轨迹的列表，每个轨迹是一个 NumPy 数组。
        output_folder (str): 保存图像的文件夹路径。
        filename (str): 当前文件的名字，用于命名输出图像。
    """
    #plt.figure(figsize=(10, 8))
    plt.figure()
    for i, trajectory in enumerate(trajectories):
        if trajectory.shape[1] < 2:
            print(f"Skipping trajectory {i+1} in {filename}: insufficient dimensions.")
            continue
        x = trajectory[:, 0]  # x 坐标
        y = trajectory[:, 1]  # y 坐标
        plt.plot(x, y, label=f"Trajectory {i+1}")
    plt.xlim([0, 15])  # 设置 x 轴范围
    plt.ylim([-7.5, 7.5])  # 设置 y 轴范围

    # 图形细节设置
    # plt.title(f"Trajectories in {filename}")
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.legend()
    plt.grid()
    #plt.axis("equal")  # 保持 x 和 y 轴比例一致

    # 保存图像，文件名与输入文件名一致，但扩展名改为 .png
    image_name = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(output_folder, image_name)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    # 用户输入文件夹路径
    #folder_path = input("Please enter the folder path containing pickle files: ").strip()
    folder_path = '/home/zhoutong/dir_sda/betty/zt_mp_lib/data/raw_data/traj_png/'
    input_path = '/home/zhoutong/dir_sda/betty/zt_mp_lib/data/raw_data/traj/'
    #input_path = '/home/zhoutong/dir_sda/betty/zt_mp_lib/data/raw_data/path/circle_path/'
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # 加载轨迹数据
    print("Loading pickle files...")
    trajectories_dict = load_pickle_files(input_path)

    if not trajectories_dict:
        print("No valid pickle files found in the folder.")
        return

    # 创建输出文件夹
    output_folder = os.path.join(folder_path, "output_plots")
    output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)

    # 为每个文件绘制轨迹图
    for filename, trajectories in trajectories_dict.items():
        print(f"Plotting trajectories from {filename}...")
        plot_trajectories(trajectories, output_folder, filename)

    print("All plots have been saved.")


# 调用主函数
if __name__ == "__main__":
    main()