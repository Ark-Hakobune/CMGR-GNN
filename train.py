# train_bimodal_age.py
from __future__ import annotations
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import data_loader
import params
from utils import select_top_nodes

from graphinit import (
    batch_corr_np_to_data_list,
    batch_signed_corr_np_to_data_list,
    data_list_to_batch,
    build_mask_vec_from_other_adj_bnn,
)
from model import FuncMaskGuidedGCNEncoder,BiModalTransformerFusion, DualModalityMaskedEncoder

from segmented_training import SegmentedAgePredictor, SegmentedLoss, create_segment_data_splits

import os
import matplotlib
matplotlib.use("Agg")  # SSH/无显示环境安全渲染
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def extract_attention_scores(encoder, fusion, loader, cfg, age_groups=None):
    """提取每个样本的、每个脑区的独立贡献（梯度 L2 范数）。

    返回一个样本列表（list），每个元素为 dict，包含：
      - age: float
      - age_group: str
      - global_idx
      - grad_node_func: (68,) ndarray, 功能模态各脑区的贡献
      - grad_node_morph: (68,) ndarray, 形态模态各脑区的贡献
    """
    if age_groups is None:
        age_groups = [(45, 135), (135, 225), (225, 315), (315, 450), (450, 630), (630, 810)]

    samples = []
    encoder.eval()
    fusion.eval()
    device = next(encoder.parameters()).device
    print(f"Model device: {device}")

    global_subject_idx = 0

    for batch in loader:
        corr_morph_bnn, corr_func_bnn, ages = batch
        if isinstance(ages, np.ndarray):
            ages = torch.from_numpy(ages).float()
        ages = ages.to(device).view(-1).float()

        func_batch, morph_batch = build_batches_from_corr(
            corr_func_bnn, corr_morph_bnn, cfg
        )

        # 确保 GCN 编码器记录了节点特征
        encoder.main_encoder.last_node_func = None
        encoder.main_encoder.last_node_morph = None

        # 前向传播
        zf, zm = encoder(func_batch, morph_batch)
        y_pred = fusion(zf, zm)

        # 检查是否成功获取了节点特征
        node_feat_func = encoder.main_encoder.last_node_func
        node_feat_morph = encoder.main_encoder.last_node_morph

        if node_feat_func is None or node_feat_morph is None:
            print("[WARN] Could not retrieve node-level features from encoder. Skipping gradient calculation.")
            continue

        B = ages.size(0)
        num_nodes = func_batch.x.size(0) // B  # 假设每个样本的节点数相同

        for i in range(B):
            age = float(ages[i].cpu().item())

            # 计算输出相对于节点特征的梯度
            grads = torch.autograd.grad(
                outputs=y_pred[i],
                inputs=[node_feat_func, node_feat_morph],
                retain_graph=True,
                allow_unused=True,
            )

            grad_node_func_all, grad_node_morph_all = grads
            
            # 提取当前样本的节点梯度 [num_nodes, C]
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            
            grad_func_i = grad_node_func_all[start_idx:end_idx] if grad_node_func_all is not None else torch.zeros(num_nodes, 1, device=device)
            grad_morph_i = grad_node_morph_all[start_idx:end_idx] if grad_node_morph_all is not None else torch.zeros(num_nodes, 1, device=device)

            # 对每个节点的梯度向量计算 L2 范数，得到 (68,) 的重要性分数
            grad_node_func = torch.linalg.norm(grad_func_i, ord=2, dim=1).cpu().numpy()
            grad_node_morph = torch.linalg.norm(grad_morph_i, ord=2, dim=1).cpu().numpy()

            group_name = "out_of_range"
            if age_groups:
                # Check defined ranges
                for start, end in age_groups:
                    if start <= age < end:
                        group_name = f"{start}-{end}"
                        break
                
                # If not found, check for the final open-ended group
                if group_name == "out_of_range":
                    last_boundary = age_groups[-1][1]
                    if age >= last_boundary:
                        group_name = f"{last_boundary}-inf"
            
            samples.append({
                "global_idx": int(global_subject_idx),
                "age": float(age),
                "age_group": group_name,
                "grad_node_func": grad_node_func,
                "grad_node_morph": grad_node_morph,
            })
            global_subject_idx += 1

    return samples


def extract_saliency_by_perturbation(encoder, fusion, loader, cfg, age_groups=None):
    """对每个样本对 68 个节点逐一做扰动：用其他节点的均值替换该节点的特征，
    计算原预测与扰动后预测之差的绝对值作为 Saliency Index。

    返回样本列表，每个元素包含：
      - age, age_group, global_idx
      - saliency_perturb_func: (68,) ndarray
      - saliency_perturb_morph: (68,) ndarray
    """
    if age_groups is None:
        age_groups = [(45, 135), (135, 225), (225, 315), (315, 450), (450, 630), (630, 810)]

    samples = []
    encoder.eval(); fusion.eval()
    device = next(encoder.parameters()).device

    global_subject_idx = 0

    for batch in loader:
        corr_morph_bnn, corr_func_bnn, ages = batch
        if isinstance(ages, np.ndarray):
            ages = torch.from_numpy(ages).float()
        ages = ages.to(device).view(-1).float()

        func_batch, morph_batch = build_batches_from_corr(
            corr_func_bnn, corr_morph_bnn, cfg
        )

        B = ages.size(0)
        # 原始预测
        with torch.no_grad():
            zf, zm = encoder(func_batch, morph_batch)
            y_pred = fusion(zf, zm).view(-1)

        num_nodes = func_batch.x.size(0) // B

        # 保存原始 x，以便恢复
        orig_fx = func_batch.x.detach().clone()
        orig_mx = morph_batch.x.detach().clone()

        for i in range(B):
            age = float(ages[i].cpu().item())
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes

            orig_pred = float(y_pred[i].detach().cpu().item())

            sal_func = np.zeros(num_nodes, dtype=float)
            sal_morph = np.zeros(num_nodes, dtype=float)

            # 准备 block 的拷贝（在 device 上）
            for j in range(num_nodes):
                # Perturb func modality
                fb_x = orig_fx.clone()
                block = fb_x[start_idx:end_idx]
                if num_nodes > 1:
                    # exclude j
                    total = block.sum(dim=0)
                    mean_other = (total - block[j]) / float(max(1, num_nodes - 1))
                else:
                    mean_other = block[j]
                block[j] = mean_other
                func_batch.x = fb_x.to(cfg.device)

                with torch.no_grad():
                    zf_p, zm_p = encoder(func_batch, morph_batch)
                    y_p = fusion(zf_p, zm_p).view(-1)
                sal_func[j] = float(abs(orig_pred - float(y_p[i].detach().cpu().item())))

                # Perturb morph modality
                mb_x = orig_mx.clone()
                block_m = mb_x[start_idx:end_idx]
                if num_nodes > 1:
                    total_m = block_m.sum(dim=0)
                    mean_other_m = (total_m - block_m[j]) / float(max(1, num_nodes - 1))
                else:
                    mean_other_m = block_m[j]
                block_m[j] = mean_other_m
                morph_batch.x = mb_x.to(cfg.device)

                with torch.no_grad():
                    zf_p2, zm_p2 = encoder(func_batch, morph_batch)
                    y_p2 = fusion(zf_p2, zm_p2).view(-1)
                sal_morph[j] = float(abs(orig_pred - float(y_p2[i].detach().cpu().item())))

                # restore func_batch.x and morph_batch.x (we use clones each iteration so it's fine)

            # 年龄段分组名
            group_name = "out_of_range"
            if age_groups:
                for start, end in age_groups:
                    if start <= age < end:
                        group_name = f"{start}-{end}"
                        break
                if group_name == "out_of_range":
                    last_boundary = age_groups[-1][1]
                    if age >= last_boundary:
                        group_name = f"{last_boundary}-inf"

            samples.append({
                "global_idx": int(global_subject_idx),
                "age": float(age),
                "age_group": group_name,
                "saliency_perturb_func": sal_func,
                "saliency_perturb_morph": sal_morph,
            })
            global_subject_idx += 1

        # 恢复 x
        func_batch.x = orig_fx.to(cfg.device)
        morph_batch.x = orig_mx.to(cfg.device)

    return samples


def extract_saliency_by_perturbation_segmented(predictor, loader, cfg, age_groups=None):
    if age_groups is None:
        age_groups = [(45, 135), (135, 225), (225, 315), (315, 450), (450, 630), (630, 810)]

    samples = []
    predictor.eval()
    device = next(predictor.parameters()).device

    global_subject_idx = 0

    for batch in loader:
        corr_morph_bnn, corr_func_bnn, ages = batch
        if isinstance(ages, np.ndarray):
            ages = torch.from_numpy(ages).float()
        ages = ages.to(device).view(-1).float()

        func_batch, morph_batch = build_batches_from_corr(
            corr_func_bnn, corr_morph_bnn, cfg
        )

        B = ages.size(0)
        with torch.no_grad():
            y_pred, _ = predictor(func_batch, morph_batch, ages, use_segment_heads=True)
            y_pred = y_pred.view(-1)

        num_nodes = func_batch.x.size(0) // B
        orig_fx = func_batch.x.detach().clone()
        orig_mx = morph_batch.x.detach().clone()

        for i in range(B):
            age = float(ages[i].cpu().item())
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            orig_pred = float(y_pred[i].detach().cpu().item())

            sal_func = np.zeros(num_nodes, dtype=float)
            sal_morph = np.zeros(num_nodes, dtype=float)

            for j in range(num_nodes):
                fb_x = orig_fx.clone()
                block = fb_x[start_idx:end_idx]
                if num_nodes > 1:
                    total = block.sum(dim=0)
                    mean_other = (total - block[j]) / float(max(1, num_nodes - 1))
                else:
                    mean_other = block[j]
                block[j] = mean_other
                func_batch.x = fb_x.to(cfg.device)

                with torch.no_grad():
                    y_p, _ = predictor(func_batch, morph_batch, ages, use_segment_heads=True)
                    y_p = y_p.view(-1)
                sal_func[j] = float(abs(orig_pred - float(y_p[i].detach().cpu().item())))

                mb_x = orig_mx.clone()
                block_m = mb_x[start_idx:end_idx]
                if num_nodes > 1:
                    total_m = block_m.sum(dim=0)
                    mean_other_m = (total_m - block_m[j]) / float(max(1, num_nodes - 1))
                else:
                    mean_other_m = block_m[j]
                block_m[j] = mean_other_m
                morph_batch.x = mb_x.to(cfg.device)

                with torch.no_grad():
                    y_p2, _ = predictor(func_batch, morph_batch, ages, use_segment_heads=True)
                    y_p2 = y_p2.view(-1)
                sal_morph[j] = float(abs(orig_pred - float(y_p2[i].detach().cpu().item())))

            group_name = "out_of_range"
            if age_groups:
                for start, end in age_groups:
                    if start <= age < end:
                        group_name = f"{start}-{end}"
                        break
                if group_name == "out_of_range":
                    last_boundary = age_groups[-1][1]
                    if age >= last_boundary:
                        group_name = f"{last_boundary}-inf"

            samples.append({
                "global_idx": int(global_subject_idx),
                "age": float(age),
                "age_group": group_name,
                "saliency_perturb_func": sal_func,
                "saliency_perturb_morph": sal_morph,
            })
            global_subject_idx += 1

        func_batch.x = orig_fx.to(cfg.device)
        morph_batch.x = orig_mx.to(cfg.device)

    return samples

def extract_attention_scores_segmented(predictor, loader, cfg, age_groups=None):
    """从分段模型中提取每个样本的、每个脑区的独立贡献（梯度 L2 范数）。"""
    if age_groups is None:
        age_groups = [(45, 135), (135, 225), (225, 315), (315, 450), (450, 630), (630, 810)]

    samples = []
    predictor.eval()
    device = next(predictor.parameters()).device
    print(f"Model device: {device}")

    global_subject_idx = 0

    for batch in loader:
        corr_morph_bnn, corr_func_bnn, ages = batch
        if isinstance(ages, np.ndarray):
            ages = torch.from_numpy(ages).float()
        ages = ages.to(device).view(-1).float()

        func_batch, morph_batch = build_batches_from_corr(
            corr_func_bnn, corr_morph_bnn, cfg
        )

        # 清空GCN编码器记录的节点特征
        predictor.encoder.main_encoder.last_node_func = None
        predictor.encoder.main_encoder.last_node_morph = None

        # 前向传播，会填充 last_node_* 特征
        y_pred, _ = predictor(func_batch, morph_batch, ages, use_segment_heads=True)

        # 获取GCN层之后的节点特征
        node_feat_func = predictor.encoder.main_encoder.last_node_func
        node_feat_morph = predictor.encoder.main_encoder.last_node_morph

        if node_feat_func is None or node_feat_morph is None:
            print("[WARN] Could not retrieve node-level features from encoder. Skipping gradient calculation.")
            continue

        B = ages.size(0)
        num_nodes = func_batch.x.size(0) // B

        for i in range(B):
            age = float(ages[i].cpu().item())

            grads = torch.autograd.grad(
                outputs=y_pred[i],
                inputs=[node_feat_func, node_feat_morph],
                retain_graph=True,
                allow_unused=True,
            )
            grad_node_func_all, grad_node_morph_all = grads
            
            start_idx = i * num_nodes
            end_idx = (i + 1) * num_nodes
            
            grad_func_i = grad_node_func_all[start_idx:end_idx] if grad_node_func_all is not None else torch.zeros(num_nodes, 1, device=device)
            grad_morph_i = grad_node_morph_all[start_idx:end_idx] if grad_node_morph_all is not None else torch.zeros(num_nodes, 1, device=device)

            grad_node_func = torch.linalg.norm(grad_func_i, ord=2, dim=1).cpu().numpy()
            grad_node_morph = torch.linalg.norm(grad_morph_i, ord=2, dim=1).cpu().numpy()

            group_name = "out_of_range"
            if age_groups:
                for start, end in age_groups:
                    if start <= age < end:
                        group_name = f"{start}-{end}"
                        break
                if group_name == "out_of_range":
                    last_boundary = age_groups[-1][1]
                    if age >= last_boundary:
                        group_name = f"{last_boundary}-inf"
            
            samples.append({
                "global_idx": int(global_subject_idx),
                "age": float(age),
                "age_group": group_name,
                "grad_node_func": grad_node_func,
                "grad_node_morph": grad_node_morph,
            })
            global_subject_idx += 1

    return samples

# -----------------------
# 配置容器
# -----------------------
class TrainConfig:
    def __init__(
        self,
        device: str = "cuda",
        use_signed: bool = False,                    # 是否用正/负两路图
        sparsify_kwargs: dict | None = None,         # 构图稀疏参数
        feat_kwargs: dict | None = None,             # 节点特征构建参数
        gcn_hidden: int = 64,
        gcn_layers: int = 3,
        gcn_embed: int = 128,
        gcn_share_backbone: bool = False,
        xattn_layers: int = 2,
        xattn_heads: int = 8,
        xattn_dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        grad_clip: float | None = 5.0,
        log_interval: int = 10,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.use_signed = use_signed
        self.sparsify_kwargs = sparsify_kwargs or dict(method="density", density=0.2, use_abs=True, keep_sign=False)
        # Signed 模式下，内部会用 corr_to_signed_adjs_np，忽略 keep_sign
        self.feat_kwargs = feat_kwargs or dict(k_pe=8, normalize_weight=True, use_abs_for_metrics=True)

        self.gcn_hidden = gcn_hidden
        self.gcn_layers = gcn_layers
        self.gcn_embed = gcn_embed
        self.gcn_share_backbone = gcn_share_backbone

        self.xattn_layers = xattn_layers
        self.xattn_heads = xattn_heads
        self.xattn_dropout = xattn_dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval


# -----------------------
# 构图：从 (B,N,N) → PyG Batch
# -----------------------
def build_batches_from_corr(
    corr_func_bnn: np.ndarray,
    corr_morph_bnn: np.ndarray,
    cfg: TrainConfig,
) -> tuple[Batch, Batch]:
    """把两模态相关矩阵批量转为 PyG Batch。"""
    import torch

    # 接受 numpy 或 torch 输入，统一转为 torch.Tensor
    def _ensure_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, torch.Tensor):
            return x.float()
        # 尝试直接转换
        return torch.tensor(x, dtype=torch.float32)

    corr_func_t = _ensure_tensor(corr_func_bnn)
    corr_morph_t = _ensure_tensor(corr_morph_bnn)

    # 构造 sparsify kwargs：允许 cfg 中的配置覆盖默认，但针对 signed 模式要确保使用正确的 key
    if cfg.use_signed:
        base = dict(cfg.sparsify_kwargs or {})
        # Only pass kwargs supported by corr_to_signed_adjs_np
        allowed = {"method", "density", "thr", "k", "share_support_by_abs", "zero_diagonal"}
        signed_kwargs = {}
        # Map common unsigned keys to signed equivalents when appropriate
        if "method" in base:
            signed_kwargs["method"] = base["method"]
        else:
            signed_kwargs["method"] = "density"
        signed_kwargs["density"] = base.get("density", 0.2)
        signed_kwargs["thr"] = base.get("thr", 0.3)
        signed_kwargs["k"] = base.get("k", 10)
        # share_support_by_abs can be driven by 'share_support_by_abs' or (legacy) 'use_abs'
        if "share_support_by_abs" in base:
            signed_kwargs["share_support_by_abs"] = base["share_support_by_abs"]
        elif "use_abs" in base:
            signed_kwargs["share_support_by_abs"] = bool(base["use_abs"])
        else:
            signed_kwargs["share_support_by_abs"] = True
        # optional zero_diagonal
        if "zero_diagonal" in base:
            signed_kwargs["zero_diagonal"] = base["zero_diagonal"]

        func_list = batch_signed_corr_np_to_data_list(
            corr_func_t,
            sparsify_kwargs=signed_kwargs,
            feat_kwargs=cfg.feat_kwargs,
        )
        morph_list = batch_signed_corr_np_to_data_list(
            corr_morph_t,
            sparsify_kwargs=signed_kwargs,
            feat_kwargs=cfg.feat_kwargs,
        )
    else:
        # 非 signed 模式：直接传 cfg.sparsify_kwargs（允许用户在 cfg 中定制）
        func_list = batch_corr_np_to_data_list(
            corr_func_t, sparsify_kwargs=cfg.sparsify_kwargs, feat_kwargs=cfg.feat_kwargs
        )
        morph_list = batch_corr_np_to_data_list(
            corr_morph_t, sparsify_kwargs=cfg.sparsify_kwargs, feat_kwargs=cfg.feat_kwargs
        )

    func_batch = data_list_to_batch(func_list).to(cfg.device)
    morph_batch = data_list_to_batch(morph_list).to(cfg.device)
    return func_batch, morph_batch


# -----------------------
# 延迟初始化模型（根据首个 batch 的特征维度）
# -----------------------
def build_models_lazy(func_batch: Batch, morph_batch: Batch, cfg: TrainConfig):
    in_f = int(func_batch.x.size(1))
    in_m = int(morph_batch.x.size(1))
    
    # 使用新的双模态编码器
    encoder = DualModalityMaskedEncoder(
        in_channels_func=in_f,
        in_channels_morph=in_m,
        hidden_channels=cfg.gcn_hidden,
        num_layers=cfg.gcn_layers,
        out_channels=cfg.gcn_embed,
        dropout=0.2,
        share_backbone=cfg.gcn_share_backbone,
        use_signed=cfg.use_signed,
    ).to(cfg.device).float()

    fusion = BiModalTransformerFusion(
        d_model=cfg.gcn_embed,
        nhead=cfg.xattn_heads,
        dim_ff=4 * cfg.gcn_embed,
        num_layers=cfg.xattn_layers,
        dropout=cfg.xattn_dropout,
        share_layers=False,
        pool="mean",
        out_dim=1,
    ).to(cfg.device).float()

    return encoder, fusion


@torch.no_grad()
def fit_dataset_feature_stats(train_loader, cfg) -> dict:
    """
    返回 {"mean": np.ndarray[C], "std": np.ndarray[C]} —— 只用训练集计算。
    要求：构图函数能临时关闭图内 z-score (node_features_from_adj_np 在 norm_mode 控制）。
    """
    import numpy as np
    from graphinit import batch_corr_np_to_data_list, batch_signed_corr_np_to_data_list
    from torch_geometric.data import Batch

    sums = None
    sumsqs = None
    count = 0

    for msn, fc, _age in train_loader:              # 你的 DataLoader 顺序：MSN, FC, AGE
        # 构图时把 norm_mode="graph" 保持默认，但在你的特征函数内部可先输出“未zscore”的 X。
        feat_kwargs = dict(k_pe=cfg.feat_kwargs.get("k_pe", 8),
                           normalize_weight=True,
                           use_abs_for_metrics=True,
                           norm_mode="graph")        # 先图内？→ 这里建议你做“未规范化版”，或在函数里加开关

        if cfg.use_signed:
            func_list  = batch_signed_corr_np_to_data_list(fc,  cfg.sparsify_kwargs, feat_kwargs)
            morph_list = batch_signed_corr_np_to_data_list(msn, cfg.sparsify_kwargs, feat_kwargs)
        else:
            func_list  = batch_corr_np_to_data_list(fc,  cfg.sparsify_kwargs, feat_kwargs)
            morph_list = batch_corr_np_to_data_list(msn, cfg.sparsify_kwargs, feat_kwargs)

        # 把 Data 列表拼 Batch，然后直接拼接它们的 x（所有节点特征）
        func_batch = Batch.from_data_list(func_list)
        morph_batch = Batch.from_data_list(morph_list)

        X_f = func_batch.x.cpu().float().numpy()
        X_m = morph_batch.x.cpu().float().numpy()

        X = np.concatenate([X_f, X_m], axis=0)      # 所有节点一起估计（两模态同一维度 C）

        if sums is None:
            C = X.shape[1]
            sums   = np.zeros(C, dtype=np.float64)
            sumsqs = np.zeros(C, dtype=np.float64)

        sums   += X.sum(axis=0, dtype=np.float64)
        sumsqs += (X.astype(np.float64) ** 2).sum(axis=0)
        count  += X.shape[0]

    mean = (sums / max(1, count)).astype(np.float32)
    var  = (sumsqs / max(1, count) - mean.astype(np.float64)**2).astype(np.float32)
    std  = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return {"mean": mean, "std": std}


# -----------------------
# 单个 epoch：train / eval
# -----------------------
def _run_one_epoch(
    loader: DataLoader,
    encoder: nn.Module,
    fusion: nn.Module,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None,
    train: bool = True,
    age_edges: list[float] | None = None,
    bin_weights: torch.Tensor | None = None
):
    mode = "train" if train else "eval"
    if train:
        encoder.train(); fusion.train()
    else:
        encoder.eval(); fusion.eval()

    # 使用 reduction='none' 来获取每个样本的损失
    loss_fn = nn.SmoothL1Loss(reduction='none')
    total_loss, total_mae, total_count = 0.0, 0.0, 0

    for step, batch in enumerate(loader, start=1):
        corr_morph_bnn, corr_func_bnn, ages = batch
        if isinstance(ages, np.ndarray):
            ages = torch.from_numpy(ages).float()
        ages = ages.to(cfg.device).view(-1).float()

        # 构建batch
        func_batch, morph_batch = build_batches_from_corr(
            corr_func_bnn, corr_morph_bnn, cfg
        )

        # 使用新的编码器进行前向传播
        zf, zm = encoder(func_batch, morph_batch)
        y_pred = fusion(zf, zm)
        
        # 计算每个样本的原始损失
        per_sample_loss = loss_fn(y_pred, ages)
        
        # 计算无权重的MAE（用于监控和日志）
        mae_unweighted = (y_pred.detach() - ages).abs().mean().item()
        
        # 如果提供了权重，则应用加权损失
        if train and age_edges is not None and bin_weights is not None:
            sample_weights = get_sample_weights(ages, age_edges, bin_weights, cfg.device)
            loss = (per_sample_loss * sample_weights).mean()
        else:
            # 在评估或不加权时，直接取平均
            loss = per_sample_loss.mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(fusion.parameters()), cfg.grad_clip)
            optimizer.step()

        batch_size = ages.size(0)
        total_loss  += loss.item() * batch_size
        total_mae   += mae_unweighted * batch_size
        total_count += batch_size

        if train and (step % cfg.log_interval == 0):
            print(f"[train] step {step}/{len(loader)} loss={loss.item():.4f} mae={mae_unweighted:.4f}")

    avg_loss = total_loss / max(1, total_count)
    avg_mae  = total_mae / max(1, total_count)
    print(f"[{mode}] Loss={avg_loss:.4f} samples={total_count}")
    return avg_loss, avg_mae

# -----------------------
# 收集预测结果（不训练）    
# -----------------------
def collect_preds_on_loader(loader: DataLoader, encoder, fusion, cfg):
    """对给定 loader（训练集/验证集）收集 y_true / y_pred。"""
    encoder.eval(); fusion.eval()
    ys, ps = [], []
    for msn, fc, ages in loader:   # 你的 DataLoader 顺序是 (MSN, FC, AGE)
        ages = torch.as_tensor(ages, device=cfg.device, dtype=torch.float32).view(-1)

        func_batch, morph_batch = build_batches_from_corr(fc, msn, cfg)
        
        # 使用与训练时相同的调用方式，让模型内部自己计算注意力
        zf, zm = encoder(func_batch, morph_batch)
        y_pred = fusion(zf, zm).to(dtype=torch.float32)

        ys.append(ages.detach().cpu().numpy())
        ps.append(y_pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
    y_pred = np.concatenate(ps, axis=0).astype(np.float32, copy=False)
    return y_true, y_pred

def collect_preds_on_loader_segmented(loader: DataLoader, predictor: SegmentedAgePredictor, cfg: TrainConfig):
    """对给定 loader 收集 y_true / y_pred (分段模型)。"""
    predictor.eval()
    ys, ps = [], []
    for msn, fc, ages in loader:
        ages_tensor = torch.as_tensor(ages, device=cfg.device, dtype=torch.float32).view(-1)
        func_batch, morph_batch = build_batches_from_corr(fc, msn, cfg)
        
        with torch.no_grad():
            y_pred, _ = predictor(func_batch, morph_batch, ages_tensor, use_segment_heads=True)
            y_pred = y_pred.to(dtype=torch.float32)

        ys.append(ages_tensor.detach().cpu().numpy())
        ps.append(y_pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
    y_pred = np.concatenate(ps, axis=0).astype(np.float32, copy=False)
    return y_true, y_pred

def summarize_mae_by_age_bands(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               mode: str = "quantile",
                               edges: list[float] | None = None):
    """
    按年龄段统计 MAE。
    - mode='quantile': 自动按分位数（默认5组）划分。
    - mode='custom': 使用自定义的 'edges' 列表，并自动处理最后一个开放区间（例如 >最后一个值）。
    """
    assert y_true.shape == y_pred.shape
    y_true = y_true.astype(np.float32, copy=False)
    y_pred = y_pred.astype(np.float32, copy=False)

    if mode == "quantile":
        # Quantile mode creates 5 bands by default
        qs = np.quantile(y_true, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges = qs.tolist()
    else:  # custom mode
        assert edges is not None and len(edges) >= 2, "Custom edges require at least 2 boundaries"
        edges = sorted(edges)

    out = []
    # Create bands from the provided edges
    for i in range(len(edges) - 1):
        l, r = edges[i], edges[i+1]
        # First band is inclusive on both ends, subsequent are (left, right]
        if i == 0:
            mask = (y_true >= l) & (y_true <= r)
        else:
            mask = (y_true > l) & (y_true <= r)
        cnt = int(mask.sum())
        mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask]))) if cnt > 0 else float("nan")
        out.append({"range": (float(l), float(r)), "count": cnt, "mae": mae})

    # Now, handle the last open-ended band for custom mode
    if mode == 'custom' and edges:
        last_edge = edges[-1]
        mask = y_true > last_edge
        cnt = int(mask.sum())
        if cnt > 0:  # Only add the band if there are samples in it
            mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
            out.append({"range": (float(last_edge), float('inf')), "count": cnt, "mae": mae})
    
    # Return the original edges for consistency in reporting
    report_edges = edges if mode == 'quantile' else edges + [float('inf')]
    return out, report_edges

def print_band_table(band_stats):
    # 尝试用 pandas 漂亮展示；没有就用纯文本
    try:
        import pandas as pd
        df = pd.DataFrame([{
            "Band": f"({b['range'][0]:.2f}, {b['range'][1]:.2f}]",
            "Count": b["count"],
            "MAE": None if not np.isfinite(b["mae"]) else round(b["mae"], 3)
        } for b in band_stats])
        print("\n=== Train set MAE by age bands ===")
        print(df.to_string(index=False))
        return df
    except Exception:
        print("\n=== Train set MAE by age bands ===")
        print(f"{'Band':<24} {'Count':>6} {'MAE':>10}")
        for i, b in enumerate(band_stats, 1):
            l, r = b["range"]
            cnt, mae = b["count"], b["mae"]
            mae_str = f"{mae:.3f}" if np.isfinite(mae) else "nan"
            print(f"{i:>1}. ({l:.2f}, {r:.2f}]:{ '':<7}{cnt:>6} {mae_str:>10}")
        return None

def save_band_results(results_dir: str,
                      prefix: str,
                      y_true: np.ndarray, y_pred: np.ndarray,
                      band_stats: list[dict], edges: list[float],
                      mode: str = "quantile"):
    """保存 CSV + JSON + NPZ（y_true/y_pred/edges/mode）"""
    os.makedirs(results_dir, exist_ok=True)
    csv_path  = os.path.join(results_dir, f"{prefix}_band_mae.csv")
    json_path = os.path.join(results_dir, f"{prefix}_band_mae.json")
    npz_path  = os.path.join(results_dir, f"{prefix}_preds.npz")

    # CSV
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band_idx", "left", "right", "count", "mae"])
        for i, b in enumerate(band_stats, 1):
            l, r = b["range"]; w.writerow([i, f"{l:.6f}", f"{r:.6f}", b["count"], f"{b['mae']:.6f}"])
    # JSON（含元信息）
    summary = {
        "mode": mode,
        "edges": [float(x) for x in edges],
        "overall_mae": float(np.mean(np.abs(y_pred - y_true))),
        "overall_mse": float(np.mean((y_pred - y_true) ** 2)),
        "bands": band_stats,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # NPZ：以后无需再推理即可直接出表
    np.savez_compressed(npz_path, y_true=y_true, y_pred=y_pred, edges=np.array(edges, dtype=np.float32), mode=mode)

    print(f"\n[Saved] CSV : {csv_path}\n[Saved] JSON: {json_path}\n[Saved] NPZ : {npz_path}")
    return csv_path, json_path, npz_path

def load_and_report(results_dir: str, prefix: str):
    """离线查看已保存结果：直接加载 CSV 并打印表格、绘制柱状图。"""
    import pandas as pd

    csv_path = os.path.join(results_dir, f"{prefix}_band_mae.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] 找不到已保存的结果：{csv_path}。请先训练并保存。")
        return

    # 读取CSV
    try:
        df = pd.read_csv(csv_path)
        print("\n=== Train set MAE by age bands (from CSV) ===")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"[WARN] 读取CSV失败: {e}")
        return

    # 绘制分段MAE柱状图
    try:
        import matplotlib
        matplotlib.use('Agg')  # 兼容无GUI环境
        import matplotlib.pyplot as plt

        bands = [f"({l},{r}]" for l, r in zip(df['left'], df['right'])]
        maes = df['mae'].astype(float).tolist()
        counts = df['count'].astype(int).tolist()

        plt.figure(figsize=(7, 4))
        bars = plt.bar(bands, maes, color='#4A90E2')
        plt.xlabel('Age Band')
        plt.ylabel('MAE')
        plt.title('MAE by Age Band')
        plt.tight_layout()

        # 在柱上方标注 MAE 数值 & 样本数
        for rect, mae, cnt in zip(bars, maes, counts):
            if not (mae == mae):  # NaN
                continue
            height = rect.get_height()
            plt.annotate(f"{mae:.2f}\n(n={cnt})",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)

        out_path = os.path.join(results_dir, f"{prefix}_band_mae.png")
        plt.savefig(out_path)
        print(f"[Saved] MAE柱状图: {out_path}")
        plt.close()
    except Exception as e:
        print(f"[WARN] 绘图失败: {e}")

def plot_band_mae_bar(
    band_stats: list[dict],
    results_dir: str = "./results",
    prefix: str = "train",
    title: str | None = None,
    show_values: bool = True,
    dpi: int = 200,
):
    """
    根据 band_stats 画柱状图（X=年龄段, Y=MAE），保存为 PNG + SVG。
    band_stats 形如 [{'range': (l,r), 'count': n, 'mae': v}, ...] 共 5 段
    """
    os.makedirs(results_dir, exist_ok=True)

    ranges = [f"({b['range'][0]:.0f},{b['range'][1]:.0f}]" for b in band_stats]
    maes   = [float("nan") if b["mae"] is None else float(b["mae"]) for b in band_stats]
    counts = [b["count"] for b in band_stats]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    bars = ax.bar(ranges, maes)
    ax.set_xlabel("Age band")
    ax.set_ylabel("MAE")
    if title is None:
        title = "Train set MAE by age bands"
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=0)

    # 在柱上方标注 MAE 数值 & 样本数
    if show_values:
        for rect, mae, cnt in zip(bars, maes, counts):
            if not (mae == mae):  # NaN
                continue
            height = rect.get_height()
            ax.annotate(f"{mae:.2f}\n(n={cnt})",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    png_path = os.path.join(results_dir, f"{prefix}_band_mae.png")
    svg_path = os.path.join(results_dir, f"{prefix}_band_mae.svg")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] BAR PNG: {png_path}")
    print(f"[Saved] BAR SVG: {svg_path}")
    return png_path, svg_path


# -----------------------
# 加权损失相关辅助函数
# -----------------------
def calculate_bin_weights(loader: DataLoader, edges: list[float]) -> torch.Tensor:
    """计算每个样本的权重，用于加权损失函数。

    权重计算方法：根据样本年龄所在的区间，分配对应的权重值。
    边界示例：edges=[45, 135, 225, 315, 450, 630, 810]
    """
    all_ages = []
    for _, _, ages in loader:
        all_ages.extend(ages)
    all_ages = np.array(all_ages)

    # 计算每个区间的样本数
    counts, _ = np.histogram(all_ages, bins=edges)

    # 计算权重：可以是简单的反比重（样本数越少权重越大），也可以是其他策略
    weights = 1.0 / (counts + 1e-6)  # 加小常数防止除零
    weights = weights / weights.sum()  # 归一化为概率分布

    # 根据每个样本的年龄分配权重
    bin_indices = np.digitize(all_ages, bins=edges) - 1  # 找到每个年龄对应的区间索引
    bin_indices = np.clip(bin_indices, 0, len(weights) - 1)  # 确保索引在有效范围内
    sample_weights = weights[bin_indices]

    return torch.tensor(sample_weights, dtype=torch.float32)

def get_sample_weights(ages: torch.Tensor, edges: list[float], bin_weights: torch.Tensor, device: str) -> torch.Tensor:
    """根据年龄获取样本权重。

    ages: 样本年龄，形状 (B,)
    edges: 年龄区间边界，形状 (N,)，例如 [45, 135, 225, 315, 450, 630, 810]
    bin_weights: 每个年龄区间对应的权重，形状 (N-1,)
    device: 目标设备
    """
    B = ages.size(0)
    # 创建一个与 ages 相同形状的权重张量
    sample_weights = torch.zeros(B, device=device)

    # 遍历每个年龄区间，设置对应区间内样本的权重
    for i in range(len(edges) - 1):
        # 找到在当前区间内的样本
        if i == 0:
            mask = (ages >= edges[i]) & (ages <= edges[i+1])
        else:
            mask = (ages > edges[i]) & (ages <= edges[i+1])

        # 设置权重
        sample_weights[mask] = bin_weights[i]

    return sample_weights


def fine_tune_segment_head(
    segmented_predictor: SegmentedAgePredictor,
    train_loader,
    val_loader,
    seg_id: int,
    cfg,
    epochs: int = 10,
    lr: float | None = None,
    upsample_factor: int = 1,
):
    """
    仅微调指定段的预测头（冻结共享编码器和其他头）。

    - segmented_predictor: 已初始化并加载好共享编码器与 heads 的模型
    - train_loader/val_loader: DataLoader，batch 顺序为 (msn, fc, ages)
    - seg_id: 要微调的段索引
    - cfg: 训练配置对象（需要含 device, log_interval, grad_clip）
    - epochs: 微调轮数
    - lr: 学习率（默认使用 cfg.lr 或 1e-4）
    - upsample_factor: 对段内样本进行上采样倍数（整数），1 表示不重复

    返回：最佳 state dict 和训练/验证时间记录
    """
    device = cfg.device
    segmented_predictor.to(device)

    # 冻结编码器与除目标 head 外的参数
    for p in segmented_predictor.encoder.parameters():
        p.requires_grad = False
    for p in segmented_predictor.fusion_module.parameters():
        p.requires_grad = False
    for i, head in enumerate(segmented_predictor.segment_heads):
        if i != seg_id:
            for p in head.parameters():
                p.requires_grad = False
    for p in segmented_predictor.global_head.parameters():
        p.requires_grad = False

    # 只训练目标 head 的参数
    trainable_params = [p for p in segmented_predictor.segment_heads[seg_id].parameters() if p.requires_grad]
    if lr is None:
        lr = getattr(cfg, 'lr', 1e-4)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=getattr(cfg, 'weight_decay', 0.0))
    loss_fn = nn.SmoothL1Loss()

    best_val = float('inf')
    best_state = None
    epoch_times = []

    for epoch in range(1, epochs + 1):
        import time
        t0 = time.time()
        segmented_predictor.train()
        train_loss = 0.0
        train_count = 0

        for msn, fc, ages in train_loader:
            # ages may be numpy
            if isinstance(ages, np.ndarray):
                ages = torch.from_numpy(ages).float()
            ages = ages.to(device).view(-1)

            func_batch, morph_batch = build_batches_from_corr(fc, msn, cfg)

            # 前向并只保留目标段样本
            y_pred, segment_idx = segmented_predictor(func_batch, morph_batch, ages, use_segment_heads=True)
            mask = (segment_idx == seg_id)
            if mask.sum() == 0:
                continue

            y_true_seg = ages[mask]
            y_pred_seg = y_pred[mask]

            # 上采样：通过重复样本来放大该段的梯度（简单但有效）
            if upsample_factor > 1:
                y_true_seg = y_true_seg.repeat(upsample_factor)
                y_pred_seg = y_pred_seg.repeat(upsample_factor)

            loss = loss_fn(y_pred_seg, y_true_seg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(cfg, 'grad_clip', None) is not None:
                nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item() * y_true_seg.size(0)
            train_count += y_true_seg.size(0)

        train_avg = train_loss / max(1, train_count)
        t1 = time.time()

        # 验证
        t0v = time.time()
        segmented_predictor.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for msn, fc, ages in val_loader:
                if isinstance(ages, np.ndarray):
                    ages = torch.from_numpy(ages).float()
                ages = ages.to(device).view(-1)
                func_batch, morph_batch = build_batches_from_corr(fc, msn, cfg)
                y_pred, segment_idx = segmented_predictor(func_batch, morph_batch, ages, use_segment_heads=True)
                mask = (segment_idx == seg_id)
                if mask.sum() == 0:
                    continue
                y_true_seg = ages[mask]
                y_pred_seg = y_pred[mask]
                l = loss_fn(y_pred_seg, y_true_seg).item()
                val_loss += l * y_true_seg.size(0)
                val_count += y_true_seg.size(0)
        val_avg = val_loss / max(1, val_count)
        t1v = time.time()

        epoch_times.append({'epoch': epoch, 'train_time': t1 - t0, 'val_time': t1v - t0v, 'train_loss': train_avg, 'val_loss': val_avg})

        print(f"[fine-tune] Epoch {epoch}/{epochs} seg={seg_id} train_loss={train_avg:.4f} val_loss={val_avg:.4f} (n_train={train_count}, n_val={val_count})")

        if val_avg < best_val:
            best_val = val_avg
            best_state = {
                'seg_id': seg_id,
                'head_state': segmented_predictor.segment_heads[seg_id].state_dict(),
                'cfg': getattr(cfg, '__dict__', {}),
                'val_loss': val_avg,
                'epoch': epoch,
            }

    return best_state, epoch_times


# -----------------------
# 分段训练函数
# -----------------------
def fit_segmented(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    segment_edges: list[float] | None = None,
    segment_weights: dict[int, float] | None = None,
):
    """
    分段训练：为每个年龄段维护独立的预测头，共享编码器。
    
    参数：
    - train_loader, val_loader: 数据加载器
    - cfg: 训练配置
    - segment_edges: 年龄段边界，如 [45, 135, 225, 315, 450, 630, 810]
    - segment_weights: 各段的损失权重字典
    """
    if segment_edges is None:
        segment_edges = [45, 135, 225, 315, 450, 630, 810]
    
    if segment_weights is None:
        segment_weights = {i: 1.0 for i in range(len(segment_edges) - 1)}
    
    print(f"\n=== Starting Segmented Training ===")
    print(f"Segment edges: {segment_edges}")
    print(f"Segment weights: {segment_weights}\n")
    
    # 1. 从一个 batch 推断维度、构建基础模型
    with torch.no_grad():
        corr_func_bnn, corr_morph_bnn, ages = next(iter(train_loader))
        func_batch, morph_batch = build_batches_from_corr(corr_func_bnn, corr_morph_bnn, cfg)
    encoder, fusion = build_models_lazy(func_batch, morph_batch, cfg)
    
    # 2. 包装成分段预测器
    segmented_predictor = SegmentedAgePredictor(
        encoder=encoder,
        fusion=fusion,
        segment_edges=segment_edges,
        embed_dim=cfg.gcn_embed,
        hidden_dim=cfg.gcn_embed,
        dropout=0.3,
    ).to(cfg.device)
    
    # 3. 创建分段损失函数
    seg_loss_fn = SegmentedLoss(
        segment_weights=segment_weights,
        segment_edges=segment_edges,
        base_loss_fn=nn.SmoothL1Loss(reduction='none'),
    )
    
    # 4. 优化器和调度器
    optimizer = torch.optim.AdamW(
        segmented_predictor.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = math.inf
    best_state = None
    best_epoch = 0
    
    # 5. 训练循环
    epoch_times = []
    for epoch in range(1, cfg.max_epochs + 1):
        print(f"\n=== Segmented Epoch {epoch}/{cfg.max_epochs} ===")
        import time
        epoch_record = {"epoch": epoch, "train_time": None, "val_time": None}
        
        # 训练阶段
        t0 = time.time()
        segmented_predictor.train()
        train_loss_total = 0.0
        train_mae_by_segment = {i: 0.0 for i in range(len(segment_edges) - 1)}
        train_count_by_segment = {i: 0 for i in range(len(segment_edges) - 1)}
        
        for step, batch in enumerate(train_loader, start=1):
            corr_morph_bnn, corr_func_bnn, ages = batch
            if isinstance(ages, np.ndarray):
                ages = torch.from_numpy(ages).float()
            ages = ages.to(cfg.device).view(-1).float()
            
            func_batch, morph_batch = build_batches_from_corr(corr_func_bnn, corr_morph_bnn, cfg)
            
            # 前向传播
            y_pred, segment_idx = segmented_predictor(
                func_batch, morph_batch, ages, use_segment_heads=True
            )
            
            # 计算损失
            loss = seg_loss_fn(y_pred, ages, segment_idx)
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(segmented_predictor.parameters(), cfg.grad_clip)
            optimizer.step()
            
            # 统计
            train_loss_total += loss.item() * ages.size(0)
            
            # 按段统计 MAE
            for seg_id in range(len(segment_edges) - 1):
                mask = (segment_idx == seg_id)
                if mask.sum() > 0:
                    seg_mae = (y_pred[mask].detach() - ages[mask]).abs().mean().item()
                    train_mae_by_segment[seg_id] += seg_mae * mask.sum().item()
                    train_count_by_segment[seg_id] += mask.sum().item()
            
            if step % cfg.log_interval == 0:
                print(f"[train] step {step}/{len(train_loader)} loss={loss.item():.4f}")

        train_loss_avg = train_loss_total / max(1, sum(train_count_by_segment.values()))
        print(f"[train] Epoch {epoch} Loss={train_loss_avg:.4f}")
        print("  Train MAE by segment:")
        for seg_id in range(len(segment_edges) - 1):
            if train_count_by_segment[seg_id] > 0:
                mae = train_mae_by_segment[seg_id] / train_count_by_segment[seg_id]
                print(f"    Segment {seg_id} ({segment_edges[seg_id]:.0f}-{segment_edges[seg_id+1]:.0f}): MAE={mae:.4f} (n={train_count_by_segment[seg_id]})")
        
        t1 = time.time()
        train_elapsed = t1 - t0
        epoch_record["train_time"] = float(train_elapsed)
        print(f"Epoch {epoch} training time: {train_elapsed:.2f}s")
        epoch_times.append(epoch_record)
        
        # 验证阶段
        if val_loader is not None:
            t0v = time.time()
            segmented_predictor.eval()
            val_loss_total = 0.0
            val_mae_by_segment = {i: 0.0 for i in range(len(segment_edges) - 1)}
            val_count_by_segment = {i: 0 for i in range(len(segment_edges) - 1)}
            
            with torch.no_grad():
                for batch in val_loader:
                    corr_morph_bnn, corr_func_bnn, ages = batch
                    if isinstance(ages, np.ndarray):
                        ages = torch.from_numpy(ages).float()
                    ages = ages.to(cfg.device).view(-1).float()
                    
                    func_batch, morph_batch = build_batches_from_corr(corr_func_bnn, corr_morph_bnn, cfg)
                    
                    y_pred, segment_idx = segmented_predictor(
                        func_batch, morph_batch, ages, use_segment_heads=True
                    )
                    
                    loss = seg_loss_fn(y_pred, ages, segment_idx)
                    val_loss_total += loss.item() * ages.size(0)
                    
                    for seg_id in range(len(segment_edges) - 1):
                        mask = (segment_idx == seg_id)
                        if mask.sum() > 0:
                            seg_mae = (y_pred[mask] - ages[mask]).abs().mean().item()
                            val_mae_by_segment[seg_id] += seg_mae * mask.sum().item()
                            val_count_by_segment[seg_id] += mask.sum().item()
            
            val_loss_avg = val_loss_total / max(1, sum(val_count_by_segment.values()))
            print(f"[eval] Epoch {epoch} Loss={val_loss_avg:.4f}")
            print("  Val MAE by segment:")
            for seg_id in range(len(segment_edges) - 1):
                if val_count_by_segment[seg_id] > 0:
                    mae = val_mae_by_segment[seg_id] / val_count_by_segment[seg_id]
                    print(f"    Segment {seg_id} ({segment_edges[seg_id]:.0f}-{segment_edges[seg_id+1]:.0f}): MAE={mae:.4f} (n={val_count_by_segment[seg_id]})")
            
            scheduler.step(val_loss_avg)
            
            t1v = time.time()
            val_elapsed = t1v - t0v
            epoch_record["val_time"] = float(val_elapsed)
            print(f"Epoch {epoch} validation time: {val_elapsed:.2f}s")
            epoch_times[-1]["val_time"] = float(val_elapsed)
            
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_epoch = epoch
                best_state = {
                    "segmented_predictor": segmented_predictor.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_loss": val_loss_avg,
                    "epoch": epoch,
                    "segment_edges": segment_edges,
                }
                checkpoint_path = "./checkpoints/best_model_segmented.pth"
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save(best_state, checkpoint_path)
                print(f"[Saved] Best model to {checkpoint_path}")
    
    print(f"\n=== Segmented Training Complete ===")
    print(f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}")
    
    # 保存训练时间
    os.makedirs("./results", exist_ok=True)
    with open("./results/segmented_epoch_times.json", "w") as f:
        json.dump(epoch_times, f, indent=2)
    
    return best_state, epoch_times

# -----------------------
# 总训练入口
# -----------------------
def fit(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
):
    # 先从一个 batch 推断维度、构建模型
    with torch.no_grad():
        corr_func_bnn, corr_morph_bnn, ages = next(iter(train_loader))
        func_batch, morph_batch = build_batches_from_corr(corr_func_bnn, corr_morph_bnn, cfg)
    encoder, fusion = build_models_lazy(func_batch, morph_batch, cfg)

    # 优化器（使用 AdamW）和学习率调度器（ReduceLROnPlateau）
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(fusion.parameters()),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val = math.inf
    best_state = None
    best_epoch = 0

    epoch_times = []
    for epoch in range(1, cfg.max_epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.max_epochs} ===")
        import time
        epoch_record = {"epoch": epoch, "train_time": None, "val_time": None}

        # 训练阶段计时
        t0 = time.time()
        _ = _run_one_epoch(train_loader, encoder, fusion, cfg, optimizer, train=True, 
                           age_edges=None, bin_weights=None)
        t1 = time.time()
        train_elapsed = t1 - t0
        epoch_record["train_time"] = float(train_elapsed)
        print(f"Epoch {epoch} training time: {train_elapsed:.2f}s")
        # append epoch record for this epoch
        epoch_times.append(epoch_record)

        if val_loader is not None:
            # 验证阶段计时
            t0v = time.time()
            val_loss, val_mae = _run_one_epoch(val_loader, encoder, fusion, cfg, optimizer=None, train=False)
            t1v = time.time()
            val_elapsed = t1v - t0v
            epoch_record["val_time"] = float(val_elapsed)
            print(f"Epoch {epoch} validation time: {val_elapsed:.2f}s")
            
            # 更新学习率调度器
            scheduler.step(val_loss)

            # update the last appended record with val_time
            epoch_times[-1]["val_time"] = float(val_elapsed)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    "encoder": encoder.state_dict(),
                    "fusion": fusion.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_loss": val_loss,
                    # 保留兼容键名 'val_mse'（旧代码/检查点可能依赖），但其值现在为 val_loss
                    "val_mse": val_loss,
                    "val_mae": val_mae,
                    "epoch": epoch
                }

    if best_state is not None:
        print(f"\nBest val Loss={best_state['val_loss']:.4f} MAE={best_state['val_mae']:.4f} at epoch {best_epoch}")
        
        # 保存最佳模型
        os.makedirs("./checkpoints", exist_ok=True)
        save_path = os.path.join("./checkpoints", "best_model.pth")
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path}")

    # 保存每个 epoch 的时长信息
    try:
        os.makedirs("./results", exist_ok=True)
        import json
        times_file = os.path.join("./results", "epoch_times.json")
        # 如果存在旧文件，尝试加载并追加（保留历史）
        existing = []
        if os.path.exists(times_file):
            try:
                with open(times_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        # Append the latest run's epoch records
        if len(epoch_times) > 0:
            existing.append({
                "run_epoch_times": epoch_times
            })
        with open(times_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"Saved epoch timing info to {times_file}")
    except Exception as e:
        print(f"[WARN] Failed to save epoch timing info: {e}")
    
    # ========= 训练完成后：对训练集统计 5 段 MAE、展示表格并保存 =========
    try:
        y_tr, p_tr = collect_preds_on_loader(train_loader, encoder, fusion, cfg)

        # Custom bands as requested
        BAND_MODE = "custom"
        CUSTOM_EDGES = [45, 135, 225, 315, 450, 630, 810]

        band_stats, edges = summarize_mae_by_age_bands(
            y_tr, p_tr,
            mode=BAND_MODE,
            edges=CUSTOM_EDGES
        )

        # 3) 表格打印
        _df = print_band_table(band_stats)

        # 4) 保存结果（默认目录 ./results，可按需改为 cfg.results_dir）
        results_DIR = "./results"
        PREFIX = "train"  # 文件前缀：train_band_mae.csv / train_preds.npz / train_band_mae.json
        save_band_results(results_DIR, PREFIX, y_tr, p_tr, band_stats, edges, mode=BAND_MODE)
    except Exception as e:
        print(f"[WARN] 统计年龄段 MAE 失败：{e}")
    return encoder, fusion, best_state

cfg = TrainConfig(
    device="cuda",
    use_signed=False,
    sparsify_kwargs=dict(method="density", density=0.2, use_abs=True, keep_sign=False),
    feat_kwargs=dict(k_pe=8),
    gcn_hidden=256, gcn_layers=6, gcn_embed=512,  # Maximized model size
    xattn_layers=16, xattn_heads=8, xattn_dropout=0.3, # Increased dropout
    lr=3e-3, weight_decay=5e-3, max_epochs=20, grad_clip=5.0 # Increased weight decay
)

def save_attention_scores(scores, out_dir='./results'):
    """根据提取出的 scores 列表，按年龄段聚合后保存。

    输出文件：out_dir/aggregated_band_{start}-{end}.npz
    """
    print("Starting node contribution extraction and aggregation...")
    samples = scores  # 直接使用传入的 scores

    # 按 age_group 聚类 samples
    groups = {}
    for s in samples:
        ag = s.get('age_group', 'out_of_range')
        groups.setdefault(ag, []).append(s)

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    saved_files = []
    for ag, items in groups.items():
        count = len(items)
        print(f"Aggregating {count} samples for age band {ag}...")

        # Detect keys to aggregate: numeric arrays in sample dicts (exclude metadata keys)
        sample_keys = [k for k in items[0].keys() if k not in ('age', 'age_group', 'global_idx')]

        band_suffix = ag.replace('-', '_').replace(' ', '')

        save_dict = {'count': int(count)}

        for key in sample_keys:
            arrays = [np.asarray(it[key]) for it in items if it.get(key) is not None]
            if not arrays:
                continue
            first_shape = arrays[0].shape
            if not all(arr.shape == first_shape for arr in arrays):
                print(f"[WARN] Inconsistent shapes for key '{key}' in band {ag}. Skipping.")
                continue
            # Only aggregate 1-D or small arrays (e.g., node-level 68-length arrays)
            stacked = np.stack(arrays, axis=0)
            # mean across samples
            mean_arr = stacked.mean(axis=0)
            out_key = f"{key}_{band_suffix}"
            save_dict[out_key] = mean_arr.astype(np.float32)

        # 保存为单个 npz
        safe_ag = ag.replace(' ', '').replace(',', '_')
        out_path = os.path.join(out_dir, f"aggregated_band_{safe_ag}.npz")

        if len(save_dict) > 1:  # 除了 count 还有别的数据
            np.savez_compressed(out_path, **save_dict)
            saved_files.append(out_path)
            print(f"Saved aggregated band file: {out_path}")
        else:
            print(f"No numeric arrays to save for band {ag}.")

    print(f"Finished aggregation. {len(saved_files)} band files saved under {out_dir}")
    return saved_files

# 在主函数中修改调用方式：
if __name__ == "__main__":
    import argparse
    # --- NEW: Refresh DataLoaders with the latest batch_size from params ---
    import params
    data_loader.create_dataloaders(batch_size=params.batch_size)
    # --- END NEW ---
    
    parser = argparse.ArgumentParser()
    # 添加加载模型的参数
    parser.add_argument("--load_model", type=str, default="./checkpoints/best_model.pth",
                        help="加载已保存的模型参数文件路径")
    parser.add_argument("--extract_only", action="store_true",
                        help="仅提取注意力分数，不进行训练")
    parser.add_argument("--regenerate_report", action="store_true",
                        help="加载最佳模型，重新生成并保存训练集的 MAE 报告")
    parser.add_argument("--segmented", action="store_true",
                        help="使用分段训练模式（为每个年龄段维护独立的预测头）")
    parser.add_argument("--segment_edges", type=str, default="45,135,225,315,450,630,810",
                        help="分段边界，用逗号分隔（分段训练时使用）")
    parser.add_argument("--segment_weights", type=str, default=None,
                        help="各段损失权重，格式: 'seg0_weight,seg1_weight,...'（分段训练时使用）")
    parser.add_argument("--fine_tune_segment", type=int, default=None,
                        help="仅微调指定段的 head（传入段索引，例如 0 表示 0-90 天段）")
    parser.add_argument("--fine_tune_epochs", type=int, default=10,
                        help="微调轮数（仅在 --fine_tune_segment 时使用）")
    parser.add_argument("--fine_tune_lr", type=float, default=None,
                        help="微调学习率（仅在 --fine_tune_segment 时使用）")
    parser.add_argument("--fine_tune_upsample", type=int, default=1,
                        help="对段内样本进行上采样倍数（整数），1 表示不重复")
    # 添加报告参数
    parser.add_argument("--report_only", action="store_true",
                        help="仅展示已保存的训练集年龄段 MAE 报告，不进行训练")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="读取/保存报告的目录（默认 ./results）")
    parser.add_argument("--results_prefix", type=str, default="train",
                        help="读取/保存报告的文件前缀（默认 train）")
    # 添加注意力分数相关参数
    parser.add_argument("--attention_save_path", type=str, default="attention_scores.npz",
                        help="注意力分数保存路径")
    parser.add_argument("--age_groups", type=str, default="45,135,225,315,450,630,810",
                        help="年龄分组边界，用逗号分隔")
    parser.add_argument("--gcn_layers", type=int, default=None,
                        help="覆盖 TrainConfig.gcn_layers（如果提供）")
    parser.add_argument("--xattn_layers", type=int, default=None,
                        help="覆盖 TrainConfig.xattn_layers（如果提供）")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="覆盖 TrainConfig.max_epochs（如果提供）")
    parser.add_argument('--segmented_training', action='store_true', help='进行分段训练')
    args = parser.parse_args()

    if args.report_only:
        print("\nLoading and reporting from saved results...")
        load_and_report(args.results_dir, args.results_prefix)
    elif args.segmented_training:
        print("Running segmented training...")
        from segmented_training import perform_segmented_training
        perform_segmented_training()
    elif args.extract_only:
        print("\nLoading saved model and extracting attention scores1...")
        
        # 1. 加载保存的模型参数
        if not os.path.exists(args.load_model):
            raise FileNotFoundError(f"Model file not found: {args.load_model}")
        
        checkpoint = torch.load(args.load_model)
        
        # 检查是否为分段模型
        is_segmented = 'segmented_predictor' in checkpoint

        # 恢复配置
        cfg_dict = checkpoint.get('cfg', {})
        import inspect
        sig = inspect.signature(TrainConfig)
        valid_keys = {p.name for p in sig.parameters.values()}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        if 'device' in filtered_cfg and isinstance(filtered_cfg['device'], torch.device):
            filtered_cfg['device'] = str(filtered_cfg['device'])
        cfg = TrainConfig(**filtered_cfg)

        # CLI 参数覆盖
        if args.gcn_layers is not None:
            cfg.gcn_layers = int(args.gcn_layers)
        if args.xattn_layers is not None:
            cfg.xattn_layers = int(args.xattn_layers)
        
        # 2. 初始化模型
        msn, fc, _ = next(iter(data_loader.data_train))
        func_batch, morph_batch = build_batches_from_corr(fc, msn, cfg)
        
        if is_segmented:
            print("Segmented model detected.")
            encoder, fusion = build_models_lazy(func_batch, morph_batch, cfg)
            model_to_extract = SegmentedAgePredictor(
                encoder=encoder,
                fusion=fusion,
                segment_edges=checkpoint.get('segment_edges', [45, 135, 225, 315, 450, 630, 810]),
                embed_dim=cfg.gcn_embed,
                hidden_dim=cfg.gcn_embed
            )
            model_to_extract.to(cfg.device)  # <-- FIX: Move the entire model to the correct device
           


            model_to_extract.load_state_dict(checkpoint['segmented_predictor'], strict=False)
            print(f"Loaded segmented model from epoch {checkpoint.get('epoch', 'N/A')} with val_loss={checkpoint.get('val_loss', -1):.4f}")
            extraction_fn = extract_attention_scores_segmented
            model_args = [model_to_extract, data_loader.data_train, cfg]
        else:
            print("Standard model detected.")
            encoder, fusion = build_models_lazy(func_batch, morph_batch, cfg)
            encoder.load_state_dict(checkpoint['encoder'])
            fusion.load_state_dict(checkpoint['fusion'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')} with val_mae={checkpoint.get('val_mae', -1):.4f}")
            extraction_fn = extract_attention_scores
            model_args = [encoder, fusion, data_loader.data_train, cfg]

        # 3. 解析年龄组参数
        age_bounds = [float(x) for x in args.age_groups.split(",")]
        age_groups = [(age_bounds[i], age_bounds[i+1]) for i in range(len(age_bounds)-1)]
        
        try:
            # 4. 提取注意力分数
            os.makedirs(args.results_dir, exist_ok=True)

            print("\nAnalyzing age distribution in training data...")
            all_ages = [age for _, _, ages in data_loader.data_train for age in ages]
            print(f"Age range in training data: {np.min(all_ages):.1f} - {np.max(all_ages):.1f}")

            # Add age_groups to the function call
            scores = extraction_fn(*model_args, age_groups=age_groups)

            # Also run perturbation-based saliency experiment and combine results
            try:
                if is_segmented:
                    print("Running perturbation saliency (segmented model)...")
                    pert_scores = extract_saliency_by_perturbation_segmented(model_to_extract, data_loader.data_train, cfg, age_groups=age_groups)
                else:
                    print("Running perturbation saliency (standard model)...")
                    pert_scores = extract_saliency_by_perturbation(encoder, fusion, data_loader.data_train, cfg, age_groups=age_groups)
            except Exception as e:
                print(f"[WARN] perturbation saliency failed: {e}")
                pert_scores = []

            all_scores = []
            if isinstance(scores, list):
                all_scores.extend(scores)
            if isinstance(pert_scores, list):
                all_scores.extend(pert_scores)

            save_attention_scores(all_scores, out_dir=args.results_dir)

        except Exception as e:
            print(f"\nError during attention analysis: {e}")
            import traceback
            traceback.print_exc()
            
    elif args.regenerate_report:
        print("\nLoading model to regenerate train/test set reports...")
        
        if not os.path.exists(args.load_model):
            raise FileNotFoundError(f"Model file not found: {args.load_model}")
        
        checkpoint = torch.load(args.load_model)
        is_segmented = 'segmented_predictor' in checkpoint
        
        # --- Restore Config ---
        cfg_dict = checkpoint.get('cfg', {})
        import inspect
        sig = inspect.signature(TrainConfig)
        valid_keys = {p.name for p in sig.parameters.values()}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        if 'device' in filtered_cfg and isinstance(filtered_cfg['device'], torch.device):
            filtered_cfg['device'] = str(filtered_cfg['device'])
        cfg = TrainConfig(**filtered_cfg)

        # --- Initialize Model ---
        data_loader.create_dataloaders(batch_size=params.batch_size)
        msn_sample, fc_sample, _ = next(iter(data_loader.data_train))
        func_batch_sample, morph_batch_sample = build_batches_from_corr(fc_sample, msn_sample, cfg)
        
        if is_segmented:
            print("Segmented model detected.")
            encoder, fusion = build_models_lazy(func_batch_sample, morph_batch_sample, cfg)
            model = SegmentedAgePredictor(
                encoder=encoder,
                fusion=fusion,
                segment_edges=checkpoint.get('segment_edges', [45, 135, 225, 315, 450, 630, 810]),
                embed_dim=cfg.gcn_embed,
                hidden_dim=cfg.gcn_embed
            ).to(cfg.device)
            model.load_state_dict(checkpoint['segmented_predictor'])
            print(f"Loaded segmented model from epoch {checkpoint.get('epoch', 'N/A')} with val_loss={checkpoint.get('val_loss', -1):.4f}")
        else:
            print("Standard model detected.")
            encoder, fusion = build_models_lazy(func_batch_sample, morph_batch_sample, cfg)
            encoder.load_state_dict(checkpoint['encoder'])
            fusion.load_state_dict(checkpoint['fusion'])
            model = (encoder, fusion) # Keep as a tuple
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')} with val_mae={checkpoint.get('val_mae', -1):.4f}")

        # --- Process both train and test sets ---
        datasets_to_process = [
            ("train", data_loader.data_train),
            ("test", data_loader.data_test)
        ]

        for prefix, loader in datasets_to_process:
            print(f"\n--- Recalculating predictions on the {prefix} set ---")
            
            if is_segmented:
                y_true, y_pred = collect_preds_on_loader_segmented(loader, model, cfg)
            else:
                y_true, y_pred = collect_preds_on_loader(loader, model[0], model[1], cfg)

            print(f"\n--- Generating new MAE report for {prefix} set ---")
            BAND_MODE = "custom"
            CUSTOM_EDGES = [45, 135, 225, 315, 450, 630, 810]

            band_stats, edges = summarize_mae_by_age_bands(
                y_true, y_pred,
                mode=BAND_MODE,
                edges=CUSTOM_EDGES
            )
            _df = print_band_table(band_stats)
            save_band_results(args.results_dir, prefix, y_true, y_pred, band_stats, edges, mode=BAND_MODE)
        
        print("\nReport regeneration complete.")


    elif args.fine_tune_segment is not None:
        # 微调单个段的 head
        seg_id = int(args.fine_tune_segment)
        print(f"\n=== Fine-tune segment {seg_id} ===")

        # override cfg layers if provided
        if args.gcn_layers is not None:
            cfg.gcn_layers = int(args.gcn_layers)
        if args.xattn_layers is not None:
            cfg.xattn_layers = int(args.xattn_layers)

        # build base models
        msn_sample, fc_sample, _ = next(iter(data_loader.data_train))
        func_batch_sample, morph_batch_sample = build_batches_from_corr(fc_sample, msn_sample, cfg)
        encoder, fusion = build_models_lazy(func_batch_sample, morph_batch_sample, cfg)
        
        segmented_predictor = SegmentedAgePredictor(encoder, fusion, segment_edges=[float(x) for x in args.segment_edges.split(',')], embed_dim=cfg.gcn_embed, hidden_dim=cfg.gcn_embed, dropout=0.3)

        # try to load pretrained checkpoint if exists
        if os.path.exists("./checkpoints/best_model_segmented.pth"):
            try:
                ck = torch.load("./checkpoints/best_model_segmented.pth")

                segmented_predictor.load_state_dict(ck.get('segmented_predictor'), strict=False)
                print(f"Loaded weights from ./checkpoints/best_model_segmented.pth")
            except Exception as e:
                print(f"Warning: failed to load checkpoint ./checkpoints/best_model_segmented.pth: {e}")

        best_state, times = fine_tune_segment_head(
            segmented_predictor,
            data_loader.data_train,
            data_loader.data_test,
            seg_id=seg_id,
            cfg=cfg,
            epochs=int(args.fine_tune_epochs),
            lr=(None if args.fine_tune_lr is None else float(args.fine_tune_lr)),
            upsample_factor=int(args.fine_tune_upsample),
        )

        if best_state is not None:
            os.makedirs('./checkpoints', exist_ok=True)
            outp = f"./checkpoints/fine_tuned_segment_{seg_id}.pth"
            torch.save(best_state, outp)
            print(f"Saved fine-tuned head to {outp}")

    elif args.segmented:
        # 分段训练模式
        print("\n=== Segmented Training Mode ===")
        
        # 若通过 CLI 指定了层数，覆盖默认 cfg
        if args.gcn_layers is not None:
            cfg.gcn_layers = int(args.gcn_layers)
        if args.xattn_layers is not None:
            cfg.xattn_layers = int(args.xattn_layers)
        if args.max_epochs is not None:
            cfg.max_epochs = int(args.max_epochs)
        
        # 解析分段边界
        segment_edges = [float(x) for x in args.segment_edges.split(",")]
        print(f"Segment edges: {segment_edges}")
        
        # 解析分段权重
        segment_weights = None
        if args.segment_weights is not None:
            weights = [float(x) for x in args.segment_weights.split(",")]
            segment_weights = {i: w for i, w in enumerate(weights)}
            print(f"Segment weights: {segment_weights}")
        
        # 运行分段训练
        best_state, epoch_times = fit_segmented(
            data_loader.data_train,
            data_loader.data_test,
            cfg,
            segment_edges=segment_edges,
            segment_weights=segment_weights,
        )
        
        if best_state is not None:
            print(f"\nSegmented training complete!")
            print(f"Best model saved to ./checkpoints/best_model_segmented.pth")
    
    else:
        # 原有的训练流程
        # 若通过 CLI 指定了层数，覆盖默认 cfg
        if args.gcn_layers is not None:
            cfg.gcn_layers = int(args.gcn_layers)
        if args.xattn_layers is not None:
            cfg.xattn_layers = int(args.xattn_layers)
        if args.max_epochs is not None:
            cfg.max_epochs = int(args.max_epochs)
        encoder, fusion, best = fit(data_loader.data_train, data_loader.data_test, cfg)
        # ...rest of the training code...
