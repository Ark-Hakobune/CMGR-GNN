import os
import random
import re
import numpy as np
import pandas as pd
import torch
import params
import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

def extract_age_from_filename(filename):
    match = re.search(r'_(\d+)_', filename)
    return int(match.group(1)) if match else None


def find_age_thresholds(directory):
    ages = []
    
    # 遍历目录中的所有文件，提取年龄
    for filename in os.listdir(directory):
        age = extract_age_from_filename(filename)
        if age is not None:
            ages.append(age)
    
    # 转换为NumPy数组并排序
    ages = np.array(sorted(ages))
    
    # 计算6个分界点，使得数据均匀分布
    percentiles = np.linspace(0, 100, 6)
    thresholds = np.percentile(ages, percentiles)
    
    return thresholds


def categorize_age(age, thresholds):
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= age < thresholds[i + 1]:
            return i  # 返回范围索引
    return len(thresholds) - 2  # 最高范围索引


def categorize_batch(ages, thresholds):
    return [categorize_age(age, thresholds) for age in ages]


def matrix_power(adj_matrix, power):
    # 计算邻接矩阵的幂 adj_matrix^power
    return np.linalg.matrix_power(adj_matrix, power)


def compute_node_features(adj_matrix):
    # 使用结点与其他结点连接强度的均值作为初始特征
    features = adj_matrix.mean(dim=-1)
    return features.unsqueeze(-1)


def adj_to_edge_index_weight(adj_matrix):
    edge_index, edge_weight = pyg_utils.dense_to_sparse(adj_matrix)
    return edge_index, edge_weight


def find_cycles_using_matrix(adj_matrix):
    num_nodes = len(adj_matrix)
    rings = []
    
    # 遍历矩阵的幂次（从2开始）
    for power in range(2, num_nodes + 1):
        # 计算矩阵的第power次幂
        powered_matrix = matrix_power(adj_matrix, power)
        
        # 检查对角线元素，找出环
        for i in range(num_nodes):
            if powered_matrix[i][i] > 0:
                # 如果对角线元素大于0，说明存在一个长度为power的环
                rings.append((i, power))

    mask = torch.zeros_like(adj_matrix, dtype=torch.bool)
    for ring in rings:
        for i in range(len(ring)):
            for j in range(i + 1, len(ring)):
                node_i, node_j = ring[i], ring[j]
                mask[node_i, node_j] = True
                mask[node_j, node_i] = True

    adj_matrix[~mask] = 0
    return adj_matrix


def create_ring_adj_mask(total_nodes, rings):
    mask = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)
    for ring in rings:
        ring_len = len(ring)
        for i in range(ring_len):
            node_from = ring[i]
            node_to = ring[(i + 1) % ring_len]
            mask[node_from, node_to] = 1.0
            mask[node_to, node_from] = 1.0 

    return mask


def get_adj(adjacency_matrix, batchsize, l, r):
    # remove diagonal line
    eye_mask = torch.eye(params.node).to(params.device)
    eye_mask = eye_mask.unsqueeze(0).expand(batchsize, -1, -1)
    adjacency_matrix = adjacency_matrix * (1 - eye_mask)
    mask3 = (torch.abs(adjacency_matrix) > l) & (torch.abs(adjacency_matrix) < r)
    adj3 = adjacency_matrix.clone()
    adj3[~mask3] = 0
    edges = np.array(np.nonzero(adj3))
    edge_attr = adj3[edges[0], edges[1]]
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    
    return edge_index, edge_attr, adj3


def local_efficiency(W):
    """
    计算局部效率
    简单加权局部效率实现(O(n^3))。
    W: (n, n) 对称加权邻接矩阵
    """
    n = W.shape[0]
    E_loc = np.zeros(n)
    for i in range(n):
        # 邻域子图
        nbrs = np.where(W[i] > 0)[0]
        if len(nbrs) < 2:
            continue
        subW = W[np.ix_(nbrs, nbrs)]
        # 取倒权作距离（避免除 0），使用安全除法以避免 runtime warning
        dist = np.full_like(subW, np.inf, dtype=float)
        np.divide(1.0, subW, out=dist, where=(subW > 0))
        # Floyd-Warshall
        for k in range(len(nbrs)):
            dist = np.minimum(dist, dist[:, k][:, None] + dist[k])
        # 计算效率
            inv_dist = np.zeros_like(dist, dtype=float)
            np.divide(1.0, dist, out=inv_dist, where=(dist > 0))
        np.fill_diagonal(inv_dist, 0)
        E_loc[i] = inv_dist.sum() / (len(nbrs) * (len(nbrs) - 1))
    return E_loc


def participation_coeff(W, modules):
    """
    计算节点的参与系数
    加权 Participation Coefficient
    modules: 长度 n 的 1-D 数组，每个元素是区域所属模块编号
    """
    n = W.shape[0]
    strength = W.sum(axis=1)
    pc = np.zeros(n)
    for i in range(n):
        if strength[i] == 0:
            continue
        for m in np.unique(modules):
            idx = np.where(modules == m)[0]
            pc[i] += (W[i, idx].sum() / strength[i]) ** 2
    return 1.0 - pc


def select_top_nodes(msn_batch: torch.Tensor, top_pct=0.10, weights=(0.5, 0.5)):
    """
    根据 strength + local efficiency 选每个 MSN 前 top_pct 的节点。
    选取关注度高的节点,return mask
    ----------
    msn_batch : torch.Tensor, shape (B, n, n)
    top_pct   : float,   e.g. 0.10 表示前 10%
    weights   : tuple,   (w_strength, w_eloc) 线性权重之和 = 1
    ----------
    return    : bool tensor, shape (B, n), True 表示被选中
    """
    B, n, _ = msn_batch.shape
    k = int(torch.ceil(torch.tensor(top_pct * n)).item())
    mask = torch.zeros((B, n), dtype=torch.bool, device=msn_batch.device)

    w_str, w_eloc = weights

    for b in range(B):
        W = msn_batch[b].clone()
        W.fill_diagonal_(0)

        strength = W.sum(dim=1).cpu().numpy()
        # local efficiency 仍用 numpy 版本
        e_loc = local_efficiency(W.cpu().numpy())

        # z-score 归一化
        z = lambda x: (x - x.mean()) / (x.std() + 1e-8)
        score = w_str * z(strength) + w_eloc * z(e_loc)

        top_idx = np.argsort(score)[-k:]
        mask[b, top_idx] = True

    return mask


def threshold_graph(corr_matrix, threshold=0.5, mode='abs'):
    """
    对n*n相关性矩阵进行阈值化，构建稀疏脑网络图。
    
    参数:
        corr_matrix: ndarray (n x n) 对称矩阵，a[i][j] 是脑区i和j的相关系数
        threshold: float
            - 若 mode='abs'：为绝对值阈值，|corr| < threshold 的连接被设为0
            - 若 mode='percentile'：保留前百分之多少的连接（如0.1表示保留前10%）
        mode: str, 'abs' or 'percentile'，阈值模式
        
    返回:
        adj_matrix: ndarray, 阈值化后的邻接矩阵
        G: networkx.Graph 对象（无向加权图）
    """
    n = corr_matrix.shape[0]
    adj_matrix = np.zeros_like(corr_matrix)

    if mode == 'abs':
        # 取绝对值阈值
        mask = np.abs(corr_matrix) >= threshold
        adj_matrix[mask] = corr_matrix[mask]

    elif mode == 'percentile':
        # 去除对角线（自身相关性）
        tril = corr_matrix[np.triu_indices(n, k=1)]
        cutoff = np.percentile(np.abs(tril), 100 * (1 - threshold))
        mask = np.abs(corr_matrix) >= cutoff
        adj_matrix[mask] = corr_matrix[mask]
    
    else:
        raise ValueError("mode 应为 'abs' 或 'percentile'")

    # 构建无向图（自动去除自环）
    G = nx.from_numpy_array(adj_matrix).float()

    return adj_matrix, G


def PreForTrain_CEGCN(batch, thresholds):
    # input:[MSN, FC, AGE]
    batchsize = batch[0].shape[0]
    target = batch[2].to(params.device)            
    MSN = batch[0].to(params.device).float()
    MSN = MSN.unsqueeze(1)
    FC = batch[1].to(params.device).float()
    FC = FC.unsqueeze(1)
    batch_for_gcn = torch.repeat_interleave(torch.arange(batchsize), params.num_nodes).to(params.device)

    coarse_labels = categorize_batch(target,thresholds)
    labels_tensor = torch.tensor(coarse_labels, dtype=torch.long).to(params.device)
    _, _, FC_adj = get_adj(batch[1].to(params.device).float().view(batchsize, params.node, params.node),
                                               batchsize, 0.6, 1.0)
    _, _, MSN_adj = get_adj(batch[0].to(params.device).float().view(batchsize, params.node, params.node),
                                          batchsize, 0.6, 1.0)
    # FC_ringmask = create_ring_mask(params.node, find_cycles_using_matrix(FC_adj))
    # MSN_ringmask = create_ring_mask(params.node, find_cycles_using_matrix(MSN_adj))
    FC_CE = FC_adj #* FC_ringmask
    MSN_CE = MSN_adj #* MSN_ringmask

    FC_list, FC_CE_list, MSN_list, MSN_CE_list = [], [], [], []

    for i in range(batchsize):
        # 节点特征计算
        fc_feature = compute_node_features(FC[i])
        msn_feature = compute_node_features(MSN[i])

        # FC
        fc_edge_index, fc_weight = adj_to_edge_index_weight(FC_adj[i])
        FC_list.append(Data(x=fc_feature, edge_index=fc_edge_index, edge_attr=fc_weight))

        # FC_CE
        fc_ce_edge_index, fc_ce_weight = adj_to_edge_index_weight(FC_CE[i])
        FC_CE_list.append(Data(x=fc_feature, edge_index=fc_ce_edge_index, edge_attr=fc_ce_weight))

        # MSN
        msn_edge_index, msn_weight = adj_to_edge_index_weight(MSN_adj[i])
        MSN_list.append(Data(x=msn_feature, edge_index=msn_edge_index, edge_attr=msn_weight))

        # MSN_CE
        msn_ce_edge_index, msn_ce_weight = adj_to_edge_index_weight(MSN_CE[i])
        MSN_CE_list.append(Data(x=msn_feature, edge_index=msn_ce_edge_index, edge_attr=msn_ce_weight))

    return {
        'FC': FC_list,
        'FC_CE': FC_CE_list,
        'MSN': MSN_list,
        'MSN_CE': MSN_CE_list,
        'target': target,
        'coarse_labels': coarse_labels,
        'labels_tensor': labels_tensor,
        'batch_for_gcn': batch_for_gcn
    }


class TripletDataset():
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


def generate_triplets(adj_matrix):
    """
    根据邻接矩阵生成三元组索引。
    三元组形式为 (anchor, positive, negative)：
    - anchor 和 positive 属于同一体
    - negative 属于不同体
    """
    num_samples = len(adj_matrix)

    # 构建每个样本的同体和异体索引列表
    pos_indices = [[] for _ in range(num_samples)]
    neg_indices = [[] for _ in range(num_samples)]

    for i in range(num_samples):
        for j in range(num_samples):
            if i == j:
                continue  # 排除自身
            if adj_matrix[i][j] == 1:
                pos_indices[i].append(j)
            else:
                neg_indices[i].append(j)

    triplets = []
    for anchor in range(num_samples):
        positives = pos_indices[anchor]
        negatives = neg_indices[anchor]

        # 跳过没有正样本或负样本的情况
        if not positives or not negatives:
            continue

        valid_positives = [p for p in positives if anchor < p]
        if not valid_positives:
            continue
        sampled_positives = random.sample(valid_positives, len(valid_positives))

        sampled_negatives = random.sample(negatives, min(len(negatives), 10))

        for positive in sampled_positives:
            for negative in sampled_negatives:
                triplets.append((anchor, positive, negative))

    return triplets

def getIndiviudalFlag(temp):
    #  Indiviual[i][j]表示i与j是否属于同一个体,1代表是,0代表不是

    length = temp.__len__()
    Indiviudal = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if temp.name[i] == temp.name[j]:
                Indiviudal[i][j] = 1.
            else:
                Indiviudal[i][j] = 0.
    return torch.from_numpy(Indiviudal).to(params.device).float()