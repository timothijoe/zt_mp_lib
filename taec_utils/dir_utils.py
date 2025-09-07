import os
import pickle

def save_pickle(file_path, obj):
    """
    将Python对象序列化并保存到指定的pickle文件。


    复制
    参数:
        obj: 要序列化的Python对象。
        file_path (str): 存储pickle文件的路径。
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
    print(f"Object saved to {file_path}")
def load_pickle(file_path):
    """
    从指定的pickle文件加载Python对象。


    复制
    参数:
        file_path (str): 要加载的pickle文件的路径。
        
    返回:
        从pickle文件中加载的对象。
    """
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    #print(f"Object loaded from {file_path}")
    return obj
def ensure_directory_exists(path):
    """
    确保指定的目录存在。如果目录不存在，创建它。

    vim

    复制
    参数:
        path (str): 需要检查并可能创建的目录路径。
    """
    if path is None:
        return
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")
        
def get_files_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"path invalid: {folder_path}")
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#假设所有的路径都在这个根目录下
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)

ROOT_DATA_FOLDER = BASE_DIR + '/data'



DATA_SET_FOLDER = ROOT_DATA_FOLDER + '/dataset'
DATA_RAW_FOLDER = ROOT_DATA_FOLDER + '/raw_data'
DATA_VISUAL_FOLDER = ROOT_DATA_FOLDER + '/visual_data'

