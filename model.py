import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
import params
from data_loader import *
import torchinfo
from einops import rearrange
from torch import nn, einsum
from einops import rearrange
import math
import torchinfo
import utils
from torch_geometric.loader import DataLoader
from typing import Optional, Tuple

batch_size = params.batch_size
num_nodes = params.num_nodes_dk
device = params.device

def train_network_model(RawData, model, optimizer, criterion):
    model.train()
    early_stopping = EarlyStopping(patience=50, delta=0.001)
    for epoch in range(params.epochs):
        if epoch % 10 == 0:
            print("The epoch is:", epoch)
        loss_for_print = 0
        # stepcnt = 0
        running_loss = 0.0

        for step, batch in enumerate(RawData):
            # batch:[MSN, FC, AGE]
            optimizer.zero_grad()
            batchsize = batch[0].shape[0]
            target = batch[2].to(device)
            MSN = batch[0].to(device).float()
            MSN = MSN.unsqueeze(1)
            FC = batch[1].to(device).float()
            FC = FC.unsqueeze(1)
            out = model(FC)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            # stepcnt += 1
            running_loss += loss.item()


        print(f"Epoch {epoch+1}/{params.epochs}, Train Loss: {running_loss/len(RawData):.4f}")


        if early_stopping(running_loss, model):
            print("Early stopping triggered.")
            break
        # stepcnt = 0
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def train_CoarseToFine_model(RawData, model, optimizer, criterion):
    model.train()
    early_stopping = EarlyStopping(patience=50, delta=0.001)
    for epoch in range(params.epochs):
        if epoch % 10 == 0:
            print("The epoch is:", epoch)
        loss_for_print = 0
        # stepcnt = 0
        running_loss = 0.0
        thresholds = utils.find_age_thresholds(params.fc_path)
        coarse_criterion = nn.CrossEntropyLoss()

        for step, batch in enumerate(RawData):
            # batch:[MSN, FC, AGE]
            optimizer.zero_grad()
            FC, MSN, FC_CE, MSN_CE, target, coarse_labels, labels_tensor =utils.PreForTrain_CEGCN(batch, thresholds)
            
            coarse_logits, fine_pred = model(FC, MSN, FC_CE, MSN_CE)

            fine_pred = fine_pred.squeeze(1)
            loss_coarse = coarse_criterion(coarse_logits, labels_tensor)
            loss = criterion(fine_pred, target) + loss_coarse
            loss.backward()
            optimizer.step()
            # stepcnt += 1
            running_loss += loss.item()


        print(f"Epoch {epoch+1}/{params.epochs}, Train Loss: {running_loss/len(RawData):.4f}")


        if early_stopping(running_loss, model):
            print("Early stopping triggered.")
            break
        # stepcnt = 0
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def train_CoarseToFine_model_2505(RawData, model, optimizer, criterion):
    # 提取MSN重要节点的版本
    model.train()
    early_stopping = EarlyStopping(patience=50, delta=0.001)
    for epoch in range(params.epochs):
        if epoch % 10 == 0:
            print("The epoch is:", epoch)
        loss_for_print = 0
        # stepcnt = 0
        running_loss = 0.0
        thresholds = utils.find_age_thresholds(params.fc_path)
        coarse_criterion = nn.CrossEntropyLoss()

        for step, batch in enumerate(RawData):
            # batch:[MSN, FC, AGE]
            optimizer.zero_grad()
            FC, MSN, FC_CE, MSN_CE, target, coarse_labels, labels_tensor =utils.PreForTrain_CEGCN(batch, thresholds)
            
            coarse_logits, fine_pred = model(FC, MSN, FC_CE, MSN_CE)

            fine_pred = fine_pred.squeeze(1)
            loss_coarse = coarse_criterion(coarse_logits, labels_tensor)
            loss = criterion(fine_pred, target) + loss_coarse
            loss.backward()
            optimizer.step()
            # stepcnt += 1
            running_loss += loss.item()


        print(f"Epoch {epoch+1}/{params.epochs}, Train Loss: {running_loss/len(RawData):.4f}")


        if early_stopping(running_loss, model):
            print("Early stopping triggered.")
            break
        # stepcnt = 0
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def test_model(testData, model):
    model.eval()
    sample, true_coarse, true_fine = dataset[0]
    sample = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        coarse_logits, fine_pred = model(sample)
        predicted_coarse = torch.argmax(coarse_logits, dim=1).item()
        print("\n测试样本:")
        print("真实粗分类标签:", true_coarse.item(), "预测粗分类标签:", predicted_coarse)
        print("真实年龄:", true_fine.cpu().numpy())
        print("预测年龄:", fine_pred.cpu().numpy())

def getEdgeindex(adjacency_matrix):
    edge_indexs = []
    edge_weights = []
    for i in range(batch_size):
        adj_matrix = adjacency_matrix[i]
        edge_index = torch.nonzero(adj_matrix != 0).t().contiguous()
        edge_index = edge_index + i * num_nodes
        edge_indexs.append(edge_index.view(2, -1))

        weight = adj_matrix[edge_index[0] - i * num_nodes, edge_index[1] - i * num_nodes]
        edge_weights.append(weight)
    edge_index = torch.cat(edge_indexs, dim=1)  # 连接所有图的边索引
    edge_weight = torch.cat(edge_weights)  # 连接所有图的边权重
    return edge_index, edge_weight

def normalize(feature):
    # input: tensor(batch, dim)
    mean = feature.mean(dim=0, keepdim=True)
    std = feature.std(dim=0, keepdim=True)
    normalized_feature = (feature - mean) / (std + 1e-8)
    return normalized_feature

# def create_fine_predictors():
#     base_model = Guidence_Attn_GCN
#     fine_predictors = {i: base_model(16, 16, 16, 64, 256).to(device) for i in range(5)}
#     return fine_predictors

def gumbel_softmax_sample(logits, tau=1.0, eps=1e-10):
    # 采样 Gumbel 噪声
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)

def build_func_mask_from_morph_adj(
    morph_adj_bnn: np.ndarray,         # 形状 (B, n, n) 的形态模态邻接
    func_batch: Batch,                  # PyG 的功能图 batch(多个图串联)
    select_top_nodes,                   # 你提供的函数：ndarray[B,n,n] -> bool ndarray[B,n]
) -> torch.Tensor:
    """
    返回: torch.float32, 形状 [N_total],与 func_batch.x 的节点顺序一一对应。
    - 要求 func_batch 内第 i 个图的节点数 == morph_adj_bnn[i].shape[0] == morph_adj_bnn[i].shape[1]
    """
    assert isinstance(morph_adj_bnn, np.ndarray) and morph_adj_bnn.ndim == 3, \
        "morph_adj_bnn 必须是 ndarray[B,n,n]"

    bool_masks = select_top_nodes(morph_adj_bnn)  # (B, n), dtype=bool
    if bool_masks.dtype != np.bool_:
        bool_masks = bool_masks.astype(np.bool_)

    # PyG 的 Batch.ptr: 累积节点数,长度 B+1;两两相减是每个图的节点数
    ptr = func_batch.ptr  # shape [B+1]
    B = ptr.numel() - 1
    assert B == bool_masks.shape[0], f"Batch 内样本数 {B} 与形态邻接的批大小 {bool_masks.shape[0]} 不一致"

    parts = []
    for i in range(B):
        n_i = int(ptr[i + 1] - ptr[i])
        mask_i = bool_masks[i]
        assert mask_i.shape[0] == n_i, \
            f"第 {i} 个图节点数 {n_i} 与形态掩码长度 {mask_i.shape[0]} 不一致"
        parts.append(torch.from_numpy(mask_i.astype(np.float32)))  # 0/1 -> float
    mask_vec = torch.cat(parts, dim=0).to(func_batch.x.device)     # [N_total]
    return mask_vec


# -----------------------------
# 门控/加权基础组件
# -----------------------------
def _ensure_edge_weight(edge_index: torch.Tensor,
                        edge_weight: torch.Tensor | None,
                        device: torch.device) -> torch.Tensor:
    if edge_weight is None:
        E = edge_index.size(1)
        edge_weight = torch.ones(E, device=device)
    return edge_weight

def _node_gate_from_mask(mask_vec: torch.Tensor, alpha_param: torch.nn.Parameter) -> torch.Tensor:
    """
    mask_vec: [N] 浮点(0/1 或权重);gate = 1 + softplus(alpha) * mask
    返回 gate: [N,1]
    """
    if mask_vec.dim() == 1:
        mask_vec = mask_vec.unsqueeze(-1)  # [N,1]
    scale = F.softplus(alpha_param)       # >= 0
    return 1.0 + scale * mask_vec

def _reweight_edges(edge_index: torch.Tensor,
                    edge_weight: torch.Tensor,
                    node_gate: torch.Tensor,
                    beta_param: torch.nn.Parameter) -> torch.Tensor:
    """
    e_gate = 1 + softplus(beta) * 0.5 * (g_i + g_j)
    """
    src, dst = edge_index  # [2, E]
    g_i = node_gate[src].squeeze(-1)  # [E]
    g_j = node_gate[dst].squeeze(-1)  # [E]
    scale = F.softplus(beta_param)
    e_gate = 1.0 + scale * 0.5 * (g_i + g_j)
    return edge_weight * e_gate

def _node_gate(mask_vec, alpha):  # [N] or [N,1] -> [N,1]
    if mask_vec.dim()==1: mask_vec = mask_vec.unsqueeze(-1)
    return 1.0 + F.softplus(alpha) * mask_vec

def _edge_gate(edge_index, gate, beta, base_w):
    src, dst = edge_index
    scale = 1.0 + F.softplus(beta) * 0.5 * (gate[src].squeeze(-1)+gate[dst].squeeze(-1))
    return base_w * scale  # 保留原权重的“量级”,由 gate 放大/缩小

def _sym_norm(edge_index, edge_weight, num_nodes, eps=1e-12):
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    return edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]


def _split_signed_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor):
    """
    将带符号权重的边分成正/负两路表示，返回 (ei_pos, ew_pos, ei_neg, ew_neg)
    要求 edge_index: [2, E] (long), edge_weight: [E] (float)
    """
    if edge_index is None or edge_weight is None:
        return None, None, None, None
    # 若 edge_weight 是 Nx1, squeeze
    if edge_weight.dim() > 1:
        edge_weight = edge_weight.view(-1)
    # ensure long
    if not torch.is_floating_point(edge_weight):
        edge_weight = edge_weight.to(torch.float32)
    if not torch.is_floating_point(edge_index):
        # edge_index should be long; if it's float, cast to long
        try:
            edge_index = edge_index.to(torch.long)
        except Exception:
            pass

    pos_mask = edge_weight > 0
    neg_mask = edge_weight < 0

    if pos_mask.any():
        ei_pos = edge_index[:, pos_mask]
        ew_pos = edge_weight[pos_mask]
    else:
        ei_pos = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        ew_pos = torch.empty((0,), dtype=edge_weight.dtype, device=edge_weight.device)

    if neg_mask.any():
        ei_neg = edge_index[:, neg_mask]
        ew_neg = (-edge_weight[neg_mask]).abs()
    else:
        ei_neg = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        ew_neg = torch.empty((0,), dtype=edge_weight.dtype, device=edge_weight.device)

    return ei_pos.contiguous(), ew_pos.contiguous(), ei_neg.contiguous(), ew_neg.contiguous()


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Ensure edge_index is a LongTensor of shape [2, E].
    Accepts either [2, E] or [E, 2] input and returns [2, E] long tensor.
    """
    if edge_index is None:
        return edge_index
    if not torch.is_tensor(edge_index):
        raise TypeError(f"edge_index must be a tensor, got {type(edge_index)}")
    # If shape is [E, 2], transpose to [2, E]
    if edge_index.dim() == 2 and edge_index.shape[0] == 2:
        ei = edge_index.contiguous()
    elif edge_index.dim() == 2 and edge_index.shape[1] == 2:
        ei = edge_index.t().contiguous()
    else:
        raise ValueError(f"edge_index must be shape [2, E] or [E, 2], got {tuple(edge_index.shape)}")
    if ei.dtype != torch.long:
        ei = ei.long()
    return ei



class SpatialCrossAttention(nn.Module):

    def __init__(self, q_dim, out_dim, feature_dim=3):
        # input:(node*batchsize, GCNDim) (node*batchsize, morphdim)
        super(SpatialCrossAttention, self).__init__()

        self.to_q = nn.Linear(q_dim, out_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, out_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, out_dim, bias=False)

    def forward(self, x, morph_feature):
        if morph_feature is None:
            return x
        q = self.to_q(x)
        morph_feature = rearrange(morph_feature, "a b -> b a")
        k = self.to_k(morph_feature)
        v = self.to_v(morph_feature)

        # attention, what we cannot get enough of
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(CrossAttentionLayer, self).__init__()
        self.attn_b_from_a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layer_norm_b = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_b):
        attn_b, _ = self.attn_b_from_a(query=feat_b, key=feat_a, value=feat_a)
        feat_b = self.layer_norm_b(feat_b + attn_b)
        return feat_b
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2):
        super(CrossAttentionFusion, self).__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, feat_a, feat_b):
        # feat_a: [batch_size, seq_len_a, dim] (被参考的模态)
        for layer in self.layers:
            feat_b = layer(feat_a, feat_b)

        feat_b_pooled = feat_b.mean(dim=1)
        fused_feature = self.fc_out(feat_b_pooled)

        return fused_feature
    

class EarlyStopping:
    def __init__(self, patience=50, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0  # 记录验证集损失没有改善的次数
        self.best_loss = None  # 最佳的验证集损失
        self.early_stop = False  # 是否触发早停

    def __call__(self, val_loss, model):
        """
        判断是否停止训练
        :param val_loss: 当前的验证集损失
        :param model: 需要保存模型的网络
        :return: 如果早停条件满足则返回True,否则返回False
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存当前模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop
    
  
class CEGCN(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=6, negative_slope=0.2):
        super(CEGCN, self).__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=64)
        self.conv3 = GCNConv(in_channels=64, out_channels=out_channels)

        self.CEconv = torch.nn.ModuleList()
        self.CEconv.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.CEconv.append(GCNConv(hidden_channels, hidden_channels))
        self.CEconv.append(GCNConv(hidden_channels, out_channels))
        self.k = torch.nn.Parameter(torch.tensor(0.5))
        
    def forward(self, data1, data2, batch):
        x1, edge_index, edge_weight = data1.x, data1.edge_index, data1.edge_attr
        if edge_index is not None:
            edge_index = _normalize_edge_index(edge_index)
        x1 = self.conv1(x1, edge_index, edge_weight=edge_weight)
        x1 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x1 = self.conv3(x1, edge_index, edge_weight=edge_weight)
        x1 = global_mean_pool(x1, batch)

        x2, edge_index, edge_weight = data2.x, data2.edge_index, data2.edge_attr
        if edge_index is not None:
            edge_index = _normalize_edge_index(edge_index)
        for conv in self.CEconv[:-1]: # type: ignore
            x2 = conv(x2, edge_index, edge_weight=edge_weight)
            x2 = F.leaky_relu(x2, negative_slope=self.negative_slope)
        x2 = self.CEconv[-1](x2, edge_index, edge_weight=edge_weight)
        x2 = global_mean_pool(x2, batch)

        x = self.k * x1 + (1 - self.k) * x2
        return x


class CoarseFineNet_CEGCN(nn.Module):
    def __init__(self, num_coarse=5, fine_output_dim=1):
        super(CoarseFineNet_CEGCN, self).__init__()
        self.tau = 0.5
        outdim = 128
        self.CEGCN_fc= CEGCN(1, outdim, 32)
        self.CEGCN_msn= CEGCN(1, outdim, 32)
        # 粗分类头：输出5类
        
        self.coarse_fuse = nn.ModuleList([
            CrossAttentionFusion(outdim),
            nn.Linear(outdim, 5)
        ])
        self.fine_branches = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionFusion(outdim),
                nn.Linear(outdim, 1)
            ])
            for _ in range(num_coarse)
        ])

    def forward(self, modal1, modal2, modal1c, modal2c):
        # CEGCN
        f1 = self.CEGCN_fc(modal1, modal1c)
        f2 = self.CEGCN_msn(modal2, modal2c)

        coarse_logits = self.coarse_fuse(f1, f2)
        

        gate = gumbel_softmax_sample(coarse_logits, tau=self.tau)
        # 各细预测分支输出
        fine_outputs = []
        for branch in self.fine_branches:
            # out = branch(x, edge_index, batch_for_gcn)
            # fine_outputs.append(out.unsqueeze(1))
        # 拼接得到 (B, num_coarse, fine_output_dim)
            fine_outputs = torch.cat(fine_outputs, dim=1) # type: ignore
        
        gate = gate.unsqueeze(2)
        fine_pred = torch.sum(gate * fine_outputs, dim=1)
        
        return coarse_logits, fine_pred


class CoarseFineNet(nn.Module):
    def __init__(self, num_coarse=5, fine_output_dim=1):
        model = Guidence_Attn_GCN
        super(CoarseFineNet, self).__init__()
        self.tau = 0.5
        # 粗分类头：输出5类
        self.coarse_head = model(1,128)
        self.fine_branches = nn.ModuleList([
            model(1,128).to(device)
            for _ in range(num_coarse)
        ])

    def forward(self, x, data1, mask, batch):
        # 粗分类预测
        coarse_logits = self.coarse_head(x, data1, mask, batch)
        gate = gumbel_softmax_sample(coarse_logits, tau=self.tau)
        # 各细预测分支输出
        fine_outputs = []
        for branch in self.fine_branches:
            out = branch(x, data1, mask, batch)
            fine_outputs.append(out.unsqueeze(1))
        # 拼接得到 (B, num_coarse, fine_output_dim)
        fine_outputs = torch.cat(fine_outputs, dim=1)
        
        gate = gate.unsqueeze(2)
        fine_pred = torch.sum(gate * fine_outputs, dim=1)
        
        return coarse_logits, fine_pred


class Guidence_Attn_GCN(nn.Module):
    #soft attention方式融合
    def __init__(self, in_channels, out_channels, num_layers=6, attn_scale=1.5):
        super(Guidence_Attn_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=64)
        self.conv3 = GCNConv(in_channels=64, out_channels=out_channels)
        self.attn_scale = attn_scale

        self.k = torch.nn.Parameter(torch.tensor(0.5))
    
    @staticmethod
    def _compute_attention(mask, temp=5.0):
        # mask: (B, N)
        return torch.sigmoid(temp * (mask.float() - 0.5))
    
    def _reweight_adj(self, A, alpha):
        B, N, _ = A.shape
        # 扩展成 (B, N, N),公式:  A'_{ij} = α_i * α_j * A_{ij} * scale
        alpha_i = alpha.unsqueeze(2)          # (B, N, 1)
        alpha_j = alpha.unsqueeze(1)          # (B, 1, N)
        A_prime = A * (1 + self.attn_scale * alpha_i * alpha_j)
        # 对角线清零,避免自环重复
        idx = torch.arange(N, device=A.device)
        A_prime[:, idx, idx] = 0
        return A_prime

    def forward(self, data1, mask, batch):
        alpha = self._compute_attention(mask)

        x1, edge_index, edge_weight = data1.x, data1.edge_index, data1.edge_attr
        x1 = self.conv1(x1, edge_index, edge_weight=edge_weight)
        x1 = self.conv2(x1, edge_index, edge_weight=edge_weight)
        x1 = self.conv3(x1, edge_index, edge_weight=edge_weight)
        x1 = global_mean_pool(x1, batch)
        
        return x1


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, x, y):
        """
        x: [batch_size, dim]
        y: [batch_size, dim]
        """
        batch_size = x.size(0)
        
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        logits = torch.matmul(x, y.T) / self.temperature
        labels = torch.arange(batch_size).to(x.device)

        loss_x = F.cross_entropy(logits, labels)
        loss_y = F.cross_entropy(logits.T, labels)

        loss = (loss_x + loss_y) / 2
        return loss


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)

        return output1, output2, output3


class Longitudinal_Encoder(nn.Module):
    def __init__(self,
                 init_dim,
                 hid_dim=1000,
                 out_dim=500,
                 head_num=8,
                 layer_num=6,
                 ):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(init_dim, hid_dim), nn.BatchNorm1d(hid_dim), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.BatchNorm1d(out_dim), nn.LeakyReLU())

    def forward(self, x):
        # input node
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MaskGuidedGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=True, normalize=True)
        self.bn   = BatchNorm(out_channels)
        self.dropout = dropout

        # 可学习强度
        self.alpha_node = nn.Parameter(torch.tensor(0.2))  # 节点门控
        self.beta_edge  = nn.Parameter(torch.tensor(0.2))  # 边再加权

        self.use_res = (in_channels == out_channels)

    def forward(self,
                x: torch.Tensor,              # [N,C]
                edge_index: torch.Tensor,     # [2,E]
                edge_weight: torch.Tensor | None,  # [E] or None
                mask_vec: torch.Tensor,       # [N] or [N,1]
                ) -> torch.Tensor:
        # 1) 节点门控
        gate = _node_gate_from_mask(mask_vec, self.alpha_node)  # [N,1]
        x_gated = x * gate                                      # [N,C]

        # 2) 连边再加权
        # normalize edge_index first
        if edge_index is not None:
            edge_index = _normalize_edge_index(edge_index)
        ew = _ensure_edge_weight(edge_index, edge_weight, x.device)        # [E]
        ew = _reweight_edges(edge_index, ew, gate, self.beta_edge)         # [E]

        # 3) 图卷积 + 归一化 + 激活
        out = self.conv(x_gated, edge_index, ew)
        out = self.bn(out)
        out = F.relu(out)

        # 4) 残差 + Dropout
        if self.use_res:
            out = out + x
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class SignedMaskGuidedGCNLayer(nn.Module):
    """两路(正/负)卷积分支 + mask 强化;输出 = ReLU( Pos - Neg )"""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        kw = dict(add_self_loops=False, normalize=False, bias=False)
        self.conv_pos = GCNConv(in_ch, out_ch, **kw)
        self.conv_neg = GCNConv(in_ch, out_ch, **kw)
        self.bn = BatchNorm(out_ch)
        self.dropout = dropout
        self.alpha_node = nn.Parameter(torch.tensor(0.2))
        self.beta_edge_pos = nn.Parameter(torch.tensor(0.2))
        self.beta_edge_neg = nn.Parameter(torch.tensor(0.2))
        self.use_res = (in_ch == out_ch)

    def forward(self, x, 
                ei_pos, ew_pos,   # 正边 edge_index/edge_weight (非负)
                ei_neg, ew_neg,   # 负边 edge_index/edge_weight (非负, 来自 -A)
                mask_vec):        # [N] or [N,1],来自形态模态
        N = x.size(0)
        # Debug checks: detect NaN/Inf in inputs early
        if torch.isnan(x).any() or torch.isinf(x).any():
            print('[SignedLayer DEBUG] Input x contains NaN/Inf', 'nan', torch.isnan(x).any().item(), 'inf', torch.isinf(x).any().item())
            print(' x stats min/max/mean:', None if x.numel()==0 else (float(x.min()), float(x.max()), float(x.mean())))
        
        gate = _node_gate(mask_vec, self.alpha_node)  # [N,1]
        xg = x * gate

        # 连边再加权(按 gate 放大),然后分别做对称归一化
        # normalize edge indices (handle empty)
        if ei_pos is not None and ei_pos.numel() > 0:
            ei_pos = _normalize_edge_index(ei_pos)
        if ei_neg is not None and ei_neg.numel() > 0:
            ei_neg = _normalize_edge_index(ei_neg)
        ew_pos_g = _edge_gate(ei_pos, gate, self.beta_edge_pos, ew_pos)
        ew_neg_g = _edge_gate(ei_neg, gate, self.beta_edge_neg, ew_neg)
        if (ew_pos_g is not None) and (torch.isnan(ew_pos_g).any() or torch.isinf(ew_pos_g).any()):
            print('[SignedLayer DEBUG] ew_pos_g has NaN/Inf', 'nan', torch.isnan(ew_pos_g).any().item(), 'inf', torch.isinf(ew_pos_g).any().item())
        if (ew_neg_g is not None) and (torch.isnan(ew_neg_g).any() or torch.isinf(ew_neg_g).any()):
            print('[SignedLayer DEBUG] ew_neg_g has NaN/Inf', 'nan', torch.isnan(ew_neg_g).any().item(), 'inf', torch.isinf(ew_neg_g).any().item())
        nw_pos = _sym_norm(ei_pos, ew_pos_g, N) if (ei_pos is not None and ei_pos.numel() > 0) else ew_pos_g
        nw_neg = _sym_norm(ei_neg, ew_neg_g, N) if (ei_neg is not None and ei_neg.numel() > 0) else ew_neg_g

        if (nw_pos is not None) and (torch.isnan(nw_pos).any() or torch.isinf(nw_pos).any()):
            print('[SignedLayer DEBUG] nw_pos has NaN/Inf')
        if (nw_neg is not None) and (torch.isnan(nw_neg).any() or torch.isinf(nw_neg).any()):
            print('[SignedLayer DEBUG] nw_neg has NaN/Inf')

        # 两路图卷积并相减(抑制性邻居起“负贡献”)
        out = self.conv_pos(xg, ei_pos, nw_pos) - self.conv_neg(xg, ei_neg, nw_neg)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print('[SignedLayer DEBUG] out after conv contains NaN/Inf', 'nan', torch.isnan(out).any().item(), 'inf', torch.isinf(out).any().item())
        out = out.float()
        out = self.bn(out)
        out = F.relu(out)
        if self.use_res:
            out = out + x
        return F.dropout(out, p=self.dropout, training=self.training)


class FuncMaskGuidedGCNEncoder(nn.Module):
    """
    双向编码器（功能 & 形态），可切换 Signed/Unsigned：
      - use_signed=False: 使用 MaskGuidedGCNLayer，要求 Batch 有 edge_index/edge_attr
      - use_signed=True : 使用 SignedMaskGuidedGCNLayer，要求 Batch 有
                          pos_edge_index/pos_edge_attr & neg_edge_index/neg_edge_attr
    """
    def __init__(self,
                 in_channels_func: int,
                 in_channels_morph: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 out_channels: int = 128,
                 dropout: float = 0.2,
                 share_backbone: bool = False,
                 use_signed: bool = True):
        super().__init__()
        if share_backbone and (in_channels_func != in_channels_morph):
            raise ValueError("share_backbone=True 要求两侧输入维度一致。")
        self.use_signed = use_signed

        LayerCls = SignedMaskGuidedGCNLayer if use_signed else MaskGuidedGCNLayer

        def _make_stack(in_ch):
            dims = [in_ch] + [hidden_channels] * (num_layers - 1) + [hidden_channels]
            return nn.ModuleList([LayerCls(dims[i], dims[i+1], dropout=dropout) for i in range(num_layers)])

        self.layers_func  = _make_stack(in_channels_func)
        self.layers_morph = self.layers_func if share_backbone else _make_stack(in_channels_morph)

        self.proj_func  = nn.Linear(2 * hidden_channels, out_channels)
        self.proj_morph = nn.Linear(2 * hidden_channels, out_channels)
        self.act = nn.ReLU()

        self.last_gate_func  = None
        self.last_gate_morph = None
        # 保存每个节点在最后一层的表示（未池化），用于可视化/重要性分析
        self.last_node_func = None
        self.last_node_morph = None

    @torch.no_grad()
    def _default_batch_idx(self, x):
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    def _encode_one_side(self, batch, cross_mask, layers, proj):
        x = batch.x
        batch_idx = getattr(batch, 'batch', None)
        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.use_signed:
            ei_pos = getattr(batch, 'pos_edge_index', None)
            ew_pos = getattr(batch, 'pos_edge_attr', None)
            ei_neg = getattr(batch, 'neg_edge_index', None)
            ew_neg = getattr(batch, 'neg_edge_attr', None)

            # 若未提供 pos/neg 对，应尝试从常规 edge_index 和 edge_attr 中拆分
            if (ei_pos is None or ei_neg is None) and hasattr(batch, 'edge_index'):
                base_ei = getattr(batch, 'edge_index', None)
                base_ew = getattr(batch, 'edge_attr', None)
                if base_ei is None or base_ew is None:
                    raise ValueError('Signed mode requires either pos/neg edge fields or edge_index+edge_attr with signed weights.')
                ei_pos, ew_pos, ei_neg, ew_neg = _split_signed_edges(base_ei, base_ew)

            # Ensure tensors have correct dtypes
            if ei_pos is not None and ei_pos.numel() > 0:
                ei_pos = ei_pos.to(torch.long)
            if ei_neg is not None and ei_neg.numel() > 0:
                ei_neg = ei_neg.to(torch.long)

            # run through layers
            for layer in layers:
                x = layer(x, ei_pos, ew_pos, ei_neg, ew_neg, cross_mask)
            last_gate = _node_gate_from_mask(cross_mask, layers[-1].alpha_node)
        else:
            edge_index = batch.edge_index
            edge_weight = getattr(batch, 'edge_attr', None)
            for layer in layers:
                x = layer(x, edge_index, edge_weight, cross_mask)
            last_gate = _node_gate_from_mask(cross_mask, layers[-1].alpha_node)

        # 在池化前保存每个节点的最终表示（供后续可视化或重要性分析）
        last_node = x
        # 将表示保存到相应的属性，便于外部访问
        try:
            if proj is self.proj_func:
                self.last_node_func = last_node
            else:
                self.last_node_morph = last_node
        except Exception:
            # 保守处理：若属性不存在或比较失败，不阻塞前向
            pass

        g_mean = global_mean_pool(x, batch_idx)
        xw_sum = global_add_pool(x * last_gate, batch_idx)
        w_sum  = global_add_pool(last_gate, batch_idx).clamp_min(1e-8)
        wg_mean = xw_sum / w_sum

        z = torch.cat([g_mean, wg_mean], dim=-1)
        z = self.act(proj(z))
        return z, last_gate.detach()

    def encode_func(self, func_batch, morph_mask_vec):
        z, gate = self._encode_one_side(func_batch, morph_mask_vec, self.layers_func, self.proj_func)
        self.last_gate_func = gate
        return z

    def encode_morph(self, morph_batch, func_mask_vec):
        z, gate = self._encode_one_side(morph_batch, func_mask_vec, self.layers_morph, self.proj_morph)
        self.last_gate_morph = gate
        return z

    def forward(self, func_batch, morph_batch, mask_morph_on_func, mask_func_on_morph):
        zf = self.encode_func(func_batch,  mask_morph_on_func)
        zm = self.encode_morph(morph_batch, mask_func_on_morph)
        return zf, zm
    
    
class CrossAttnBlock(nn.Module):
    """
    单个 Cross-Attention Transformer 块：
      out = LN(x + MHA(q=x, k=ctx, v=ctx)) -> LN(out + FFN(out))
    其中 x 是 Query 的序列,ctx 是 Key/Value 的序列(可与 x 同或不同)。
    """
    def __init__(self, d_model: int, nhead: int = 8, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,              # [B, Lq, D]
        x_kv: torch.Tensor,             # [B, Lk, D]
        attn_mask: Optional[torch.Tensor] = None,      # [B, Lq, Lk] 或 [Lq, Lk]
        key_padding_mask: Optional[torch.Tensor] = None # [B, Lk],True 表示要mask掉的pad
    ) -> torch.Tensor:                  # [B, Lq, D]
        # Cross-Attn
        attn_out, _ = self.mha(
            query=x_q, key=x_kv, value=x_kv,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x_q + self.drop(attn_out))
        # FFN
        x2 = self.ffn(x)
        x = self.norm2(x + self.drop(x2))
        return x

# --------- 双向 Cross-Attn 堆叠 ---------
class BiModalCrossTransformer(nn.Module):
    """
    双向 Cross-Attn 融合：
      - A->B 块：以 A 为 Q,从 B 吸收信息,得到 A';
      - B->A 块：以 B 为 Q,从 A 吸收信息,得到 B';
    堆叠多层,每层都执行一次双向交互。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_ff: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_layers: bool = False,   # True 则两向共享同一层权重
    ):
        super().__init__()
        def make_layer():
            return nn.ModuleList([
                CrossAttnBlock(d_model, nhead, dim_ff, dropout),  # A<-B
                CrossAttnBlock(d_model, nhead, dim_ff, dropout),  # B<-A
            ])
        if share_layers:
            layer = make_layer()
            self.layers = nn.ModuleList([layer] * num_layers)
        else:
            self.layers = nn.ModuleList([make_layer() for _ in range(num_layers)])

    def forward(
        self,
        A: torch.Tensor,   # [B, D] or [B, L_a, D]
        B: torch.Tensor,   # [B, D] or [B, L_b, D]
        maskA_kpad: Optional[torch.Tensor] = None,  # [B, L_a]  True=pad
        maskB_kpad: Optional[torch.Tensor] = None,  # [B, L_b]
        attn_mask_Aq_Bk: Optional[torch.Tensor] = None,  # [B, L_a, L_b] or [L_a, L_b]
        attn_mask_Bq_Ak: Optional[torch.Tensor] = None,  # [B, L_b, L_a] or [L_b, L_a]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 若输入是图级向量 [B, D],升维成序列长度 1
        squeeze_a = False
        squeeze_b = False
        if A.dim() == 2:
            A = A.unsqueeze(1); squeeze_a = True
        if B.dim() == 2:
            B = B.unsqueeze(1); squeeze_b = True

        for (layer_ab, layer_ba) in self.layers:
            A = layer_ab(A, B, attn_mask=attn_mask_Aq_Bk, key_padding_mask=maskB_kpad)  # A<-B
            B = layer_ba(B, A, attn_mask=attn_mask_Bq_Ak, key_padding_mask=maskA_kpad)  # B<-A

        if squeeze_a:
            A = A.squeeze(1)  # [B, D]
        if squeeze_b:
            B = B.squeeze(1)  # [B, D]
        return A, B

# --------- 融合头(图级或序列池化后) ---------
class FusionHead(nn.Module):
    """
    将双向后的表示融合为一个向量：
      h = [A', B', |A'-B'|, A'*B'] -> MLP -> out_dim
    若输入是序列,[B, L, D] 将先做池化(mean 或 cls token)。
    """
    def __init__(self, d_model: int, out_dim: int = 1, hidden: int = 256, dropout: float = 0.1,
                 pool: str = "mean"):  # "mean" | "cls"
        super().__init__()
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3:  # [B, L, D]
            if self.pool == "mean":
                return X.mean(dim=1)
            elif self.pool == "cls":
                return X[:, 0, :]
            else:
                raise ValueError("pool must be 'mean' or 'cls'")
        return X  # [B, D]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a = self._pool(A).to(torch.float32)  # 强制转 float32
        b = self._pool(B).to(torch.float32)
        h = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)
        return self.mlp(h).squeeze(-1)  # 若 out_dim=1 -> [B]

# --------- 端到端封装(可直接回归年龄) ---------
class BiModalTransformerFusion(nn.Module):
    """
    输入：两模态的图网络输出(图级向量或序列),输出：融合后的预测(默认回归标量)。
    你也可以只用其中的表示 A', B' 自行做下游任务。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_ff: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_layers: bool = False,
        pool: str = "mean",
        out_dim: int = 1,
    ):
        super().__init__()
        self.bi_x = BiModalCrossTransformer(
            d_model=d_model, nhead=nhead, dim_ff=dim_ff,
            num_layers=num_layers, dropout=dropout, share_layers=share_layers
        )
        self.head = FusionHead(d_model=d_model, out_dim=out_dim, hidden=4*d_model, dropout=dropout, pool=pool)

    def forward(
        self,
        zA: torch.Tensor, zB: torch.Tensor,
        maskA_kpad: Optional[torch.Tensor] = None,
        maskB_kpad: Optional[torch.Tensor] = None,
        attn_mask_Aq_Bk: Optional[torch.Tensor] = None,
        attn_mask_Bq_Ak: Optional[torch.Tensor] = None,
        return_pair: bool = False
    ):
        A2, B2 = self.bi_x(zA, zB, maskA_kpad, maskB_kpad, attn_mask_Aq_Bk, attn_mask_Bq_Ak)
        y = self.head(A2, B2)
        return (y, A2, B2) if return_pair else y


class SimpleAttentionEncoder(nn.Module):
    """简单的注意力编码器，用于学习节点级别的注意力分数"""
    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # 基础GCN编码
        if edge_index is not None:
            edge_index = _normalize_edge_index(edge_index)
        h1 = self.conv1(x, edge_index, edge_weight)
        h = F.relu(self.norm1(h1))
        out2 = self.conv2(h, edge_index, edge_weight)
        # 输出注意力分数 [N, 1]
        attn = torch.sigmoid(self.norm2(out2))
        return attn

class DualModalityMaskedEncoder(nn.Module):
    """集成简单注意力编码器和FuncMaskGuidedGCNEncoder的双模态编码器"""
    def __init__(
        self,
        in_channels_func: int,
        in_channels_morph: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        out_channels: int = 128,
        dropout: float = 0.2,
        share_backbone: bool = False,
        use_signed: bool = False
    ):
        super().__init__()
        # 简单注意力编码器
        self.func_attn_encoder = SimpleAttentionEncoder(in_channels_func)
        self.morph_attn_encoder = SimpleAttentionEncoder(in_channels_morph)
        
        # 主编码器
        self.main_encoder = FuncMaskGuidedGCNEncoder(
            in_channels_func=in_channels_func,
            in_channels_morph=in_channels_morph,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            share_backbone=share_backbone,
            use_signed=use_signed
        )
        
        # 保存最后的注意力分数用于分析
        self.last_func_attn = None
        self.last_morph_attn = None
        
    def forward(self, func_batch, morph_batch, mask_morph_on_func=None, mask_func_on_morph=None):
        """
        Args:
            func_batch: PyG Batch, 功能网络数据
            morph_batch: PyG Batch, 形态网络数据
        Returns:
            tuple: (func_emb, morph_emb) 两个模态的嵌入
        """
        # 如果主编码器使用 signed 模式，但当前 Batch 只有 pos/neg 边字段，
        # 则合并为单一路（signed weights）供简单注意力编码器使用。
        def _ensure_combined_edges(batch):
            # Return (edge_index, edge_attr) or (None, None)
            ei_pos = getattr(batch, 'pos_edge_index', None)
            ew_pos = getattr(batch, 'pos_edge_attr', None)
            ei_neg = getattr(batch, 'neg_edge_index', None)
            ew_neg = getattr(batch, 'neg_edge_attr', None)

            if ei_pos is None and ei_neg is None:
                # maybe already has edge_index/edge_attr
                return getattr(batch, 'edge_index', None), getattr(batch, 'edge_attr', None)

            parts_ei = []
            parts_ew = []
            if ei_pos is not None and ei_pos.numel() > 0:
                parts_ei.append(ei_pos.long())
                parts_ew.append(ew_pos.float() if ew_pos is not None else torch.ones(ei_pos.size(1), device=ei_pos.device))
            if ei_neg is not None and ei_neg.numel() > 0:
                parts_ei.append(ei_neg.long())
                # For unsigned attention convs we must supply non-negative weights (use magnitudes)
                parts_ew.append((ew_neg.float() if ew_neg is not None else torch.ones(ei_neg.size(1), device=ei_neg.device)).abs())

            if len(parts_ei) == 0:
                return None, None
            edge_index = torch.cat(parts_ei, dim=1)
            edge_attr = torch.cat(parts_ew, dim=0)
            return edge_index, edge_attr

        # Prepare combined edges when using signed graphs so attention encoders get valid inputs
        if self.main_encoder.use_signed:
            func_ei, func_ew = _ensure_combined_edges(func_batch)
            morph_ei, morph_ew = _ensure_combined_edges(morph_batch)
        else:
            func_ei, func_ew = getattr(func_batch, 'edge_index', None), getattr(func_batch, 'edge_attr', None)
            morph_ei, morph_ew = getattr(morph_batch, 'edge_index', None), getattr(morph_batch, 'edge_attr', None)

        # 1. 计算各自模态的注意力分数（仅在未由外部提供 mask 时计算）
        computed_func_attn = None
        computed_morph_attn = None
        if mask_morph_on_func is None or mask_func_on_morph is None:
            computed_func_attn = self.func_attn_encoder(func_batch.x, func_ei, func_ew)
            computed_morph_attn = self.morph_attn_encoder(morph_batch.x, morph_ei, morph_ew)

        # 将外部提供的 mask（如果有）标准化为 tensor，并优先使用
        def _to_mask_tensor(m, ref_x):
            if m is None:
                return None
            if isinstance(m, torch.Tensor):
                t = m.float()
            elif isinstance(m, np.ndarray):
                t = torch.from_numpy(m).float()
            else:
                t = torch.tensor(m, dtype=torch.float32)
            return t.to(ref_x.device)

        final_mask_morph_on_func = _to_mask_tensor(mask_morph_on_func, func_batch.x) if mask_morph_on_func is not None else computed_morph_attn
        final_mask_func_on_morph = _to_mask_tensor(mask_func_on_morph, morph_batch.x) if mask_func_on_morph is not None else computed_func_attn

        # 保存注意力分数供分析（以计算得到的 attn 为准）
        if computed_func_attn is not None:
            self.last_func_attn = computed_func_attn
        if computed_morph_attn is not None:
            self.last_morph_attn = computed_morph_attn

        # 2. 使用最终 mask 进行主编码器前向
        func_emb, morph_emb = self.main_encoder(
            func_batch,
            morph_batch,
            mask_morph_on_func=final_mask_morph_on_func,
            mask_func_on_morph=final_mask_func_on_morph,
        )
        
        return func_emb, morph_emb
    
    def get_attention_scores(self):
        """获取最近一次前向传播的注意力分数"""
        return {
            'functional': self.last_func_attn.detach().cpu().numpy(),
            'morphological': self.last_morph_attn.detach().cpu().numpy()
        }