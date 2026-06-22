import torch
import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次运行结果一致
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 的优化以保证结果一致性



fc_path = 'C:\workplace\PycharmProjects\Predict_Infaint_Age\dataset\FC'
morphology_path = ['C:\workplace\PycharmProjects\Predict_Infaint_Age\dataset\morphology\\area_Desikan', 'C:\workplace\PycharmProjects\Predict_Infaint_Age\dataset\morphology\\curv_Desikan', 'C:\workplace\PycharmProjects\Predict_Infaint_Age\dataset\morphology\\myelin_Desikan', 'C:\workplace\PycharmProjects\Predict_Infaint_Age\dataset\morphology\\thickness_Desikan']
msn_path = 'C:\workplace\PycharmProjects/Predict_Infaint_Age/dataset/MSN'
bold_path = 'C:\workplace\dataset\\timeseries_mean_desikan'
dk_arrsize = (68, 68)

device = 'cuda'
epochs = 50
learnRate=1e-3
batch_size = 72
num_dates = 2000 # 最大天数1991
num_nodes_dk = 68
node = num_nodes_dk
SE = 4
set_seed(42)


