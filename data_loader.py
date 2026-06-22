import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import utils

import params

def readNodeFeature():
    return 

def custom_collate_fn(batch):
    # batch: list of tuples, each is (A, B, C)
    A_list, B_list, C_list = zip(*batch)  # 分别是 batch_size 个 A/B/C
    # 你想要 (B, C, A) 顺序
    return (
        torch.stack(B_list),  # B
        torch.stack(C_list),  # C
        torch.stack(A_list),  # A
    )


def save_dataset(dataset, file_path):
    """
    保存数据集的所有字段到文件中。

    参数:
    - dataset: PyTorch Dataset 或 Subset，包含多个字段（如 data, label, meta 等）。
    - file_path: 保存文件的路径。
    """
    saved_data = {}

    # 遍历 dataset 的每一项，提取所有字段
    for item in dataset:
        if isinstance(item, tuple):
            # 如果是 tuple，自动生成键名
            keys = [f"field_{i}" for i in range(len(item))]
            for key, value in zip(keys, item):
                if key not in saved_data:
                    saved_data[key] = []  # 初始化字段
                saved_data[key].append(value)
        elif isinstance(item, dict):
            # 如果是 dict，直接提取字段
            for key, value in item.items():
                if key not in saved_data:
                    saved_data[key] = []  # 初始化字段
                saved_data[key].append(value)
        else:
            raise ValueError("Unsupported data format. Dataset items must be tuple or dict.")


    # 将字段转为张量或其他合适的格式（例如列表）
    for key, value_list in saved_data.items():
        try:
            saved_data[key] = torch.stack(value_list)  # 转为张量
        except Exception:
            saved_data[key] = value_list  # 非张量数据保持原样

    # 保存到文件
    torch.save(saved_data, file_path)
    print(f"数据保存到 {file_path}")


class TripletDataset(Dataset):
    def __init__(self, dataset, triplet_indices):
        """
        Args:
            dataset (Dataset): 原始数据集，提供单个样本。
            triplet_indices (list of tuples): 三元组索引列表，每个元素是 (anchor, positive, negative)。
        """
        self.dataset = dataset
        self.triplet_indices = triplet_indices
    
    def __len__(self):
        return len(self.triplet_indices)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplet_indices[idx]
        anchor = self.dataset[anchor_idx]
        positive = self.dataset[positive_idx]
        negative = self.dataset[negative_idx]
        return anchor, positive, negative
            


class LoadDataset(Dataset):
    def __init__(self, MSN, FC, AGE):
        self.MSN = MSN
        self.FC = FC
        self.AGE = AGE
    def __len__(self):
        return len(self.FC)


    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.MSN[idx], self.FC[idx], self.AGE[idx]
        else:
            raise TypeError("Index should be an integer.")


def load_dataset(file_path):
    """
    从文件加载保存的数据集。
    """
    loaded_data = torch.load(file_path)
    loaded_data = LoadDataset(loaded_data['field_0'], loaded_data['field_1'], loaded_data['field_2'])


    return loaded_data

class BaseDataset(Dataset):
    def __init__(self, data_dir, FC_dir, array_size):

        self.MSN_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('FC.txt')]
        self.FC_files = [os.path.join(FC_dir, file) for file in os.listdir(FC_dir) if file.endswith('FC.txt')]
        self.array_size = array_size
        self.MSN = []
        self.FC = []
        self.AGE = []
        self.name = []
        # self.model = DateEmbedding(2000, 68*68)
        for file in self.MSN_files:
            array_data = np.loadtxt(file).astype(np.float32)  # 从txt文件中加载数组数据
            # 裁剪或填充数组到指定大小
            if array_data.shape[0] < self.array_size[0]:
                pad_width = ((0, self.array_size[0] - array_data.shape[0]), (0, 0))
                array_data = np.pad(array_data, pad_width, mode='constant')
            elif array_data.shape[0] > self.array_size[0]:
                array_data = array_data[:self.array_size[0], :]

            if array_data.shape[1] < self.array_size[1]:
                pad_width = ((0, 0), (0, self.array_size[1] - array_data.shape[1]))
                array_data = np.pad(array_data, pad_width, mode='constant')
            elif array_data.shape[1] > self.array_size[1]:
                array_data = array_data[:, :self.array_size[1]]
            self.MSN.append(array_data)
        for file in self.FC_files:
            array_data = np.loadtxt(file).astype(np.float32)  # 从txt文件中加载数组数据
            # 裁剪或填充数组到指定大小
            if array_data.shape[0] < self.array_size[0]:
                pad_width = ((0, self.array_size[0] - array_data.shape[0]), (0, 0))
                array_data = np.pad(array_data, pad_width, mode='constant')
            elif array_data.shape[0] > self.array_size[0]:
                array_data = array_data[:self.array_size[0], :]

            if array_data.shape[1] < self.array_size[1]:
                pad_width = ((0, 0), (0, self.array_size[1] - array_data.shape[1]))
                array_data = np.pad(array_data, pad_width, mode='constant')
            elif array_data.shape[1] > self.array_size[1]:
                array_data = array_data[:, :self.array_size[1]]
            self.FC.append(array_data)
        for filename in os.listdir(data_dir):
        # 将文件名按下划线分割
            parts = filename.split('_')
            
            # 检查文件名是否有足够的部分，以及第二部分是否为数字
            if len(parts) > 1 and parts[1].isdigit():
                age = int(parts[1])
                self.AGE.append(age)
            
        for file in self.MSN_files:
            name = file.split('\\')[-1].split('_')[0]
            # date = self.model(date)
            self.name.append(name)


    def __len__(self):
        return len(self.MSN)


    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.MSN[idx], self.FC[idx], self.AGE[idx]
        else:
            raise TypeError("Index should be an integer.")
    


# 声明全局变量，以便函数可以修改它们
data_train = None
data_test = None

def create_dataloaders(batch_size=None):
    """
    创建或重新创建全局的 data_train 和 data_test DataLoader 实例。
    """
    global data_train, data_test

    if batch_size is None:
        batch_size = params.batch_size

    # 确保数据集文件存在，如果不存在则创建
    train_path = "C:\\workplace\\PycharmProjects\\MultiModal_CrossPromptGCN\\train_dataset.pth"
    test_path = "C:\\workplace\\PycharmProjects\\MultiModal_CrossPromptGCN\\test_dataset.pth"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Dataset files not found, creating them now...")
        dataset = BaseDataset(params.msn_path, params.fc_path, array_size=params.dk_arrsize)
        test_ratio = 0.1
        test_size = int(len(dataset) * test_ratio)
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        save_dataset(train_dataset, train_path)
        save_dataset(test_dataset, test_path)
        print("Dataset files created.")

    train_data = load_dataset(train_path)
    test_data = load_dataset(test_path)

    # 创建 DataLoader 实例
    data_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    data_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print(f"DataLoaders created with batch_size = {batch_size}")

# 在模块首次导入时，使用 params.py 中的默认值创建一次
create_dataloaders()


