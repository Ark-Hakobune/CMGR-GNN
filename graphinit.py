# graph_builders_from_corr.py
# -*- coding: utf-8 -*-
"""
从 NumPy 的 (B,N,N) 相关性矩阵构图（PyTorch Geometric）的一站式工具集。
依赖: numpy, scipy, torch, torch_geometric
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from numpy.linalg import eigh
from utils import select_top_nodes
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, remove_self_loops, dense_to_sparse


# =========================
# 0) 通用小工具
# =========================
def _symmetrize_zero_diag(M: torch.Tensor, zero_diag: bool = True) -> torch.Tensor:
    M = 0.5 * (M + M.T)
    if zero_diag:
        M.fill_diagonal_(0.0)
    return M

def _mk_support_by_density(W: torch.Tensor, density: float) -> torch.Tensor:
    N = W.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    vals = W[iu[0], iu[1]]
    m = max(1, int(torch.floor(torch.tensor(density * vals.numel())).item()))
    if m < vals.numel():
        thr_val = torch.topk(vals, m, largest=True).values[-1]
        keep_u = (vals >= thr_val)
    else:
        keep_u = torch.ones_like(vals, dtype=torch.bool)
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask[iu[0], iu[1]] = keep_u
    mask = mask | mask.T
    return mask

def _mk_support_by_abs_threshold(W: torch.Tensor, thr: float) -> torch.Tensor:
    mask = (W >= thr)
    mask.fill_diagonal_(False)
    return mask

def _mk_support_by_topk_rows(W: torch.Tensor, k: int) -> torch.Tensor:
    N = W.shape[0]
    mask = torch.zeros((N, N), dtype=torch.bool, device=W.device)
    for i in range(N):
        row = W[i].clone()
        row[i] = float('-inf')
        if k > 0:
            idx = torch.topk(row, k).indices
            mask[i, idx] = True
    mask = mask | mask.T
    return mask

def corr_to_adj_np(
    corr: torch.Tensor,
    method: str = "density",
    density: float = 0.15,
    thr: float = 0.3,
    k: int = 10,
    use_abs: bool = True,
    keep_sign: bool = False,
    zero_diagonal: bool = True
) -> torch.Tensor:
    C = _symmetrize_zero_diag(corr, zero_diag=zero_diagonal)
    W = C.abs() if use_abs else C.clone()

    if method == "density":
        support = _mk_support_by_density(W, density)
    elif method == "absolute":
        support = _mk_support_by_abs_threshold(W, thr)
    elif method == "topk":
        support = _mk_support_by_topk_rows(W, k)
    else:
        raise ValueError("method must be one of {'density','absolute','topk'}")

    if keep_sign:
        A = C * support
    else:
        A = W * support
    A[torch.abs(A) < 1e-12] = 0.0
    return A

def corr_to_signed_adjs_np(
    corr: torch.Tensor,
    method: str = "density",
    density: float = 0.15,
    thr: float = 0.3,
    k: int = 10,
    share_support_by_abs: bool = True,
    zero_diagonal: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    C = _symmetrize_zero_diag(corr, zero_diag=zero_diagonal)
    Wabs = C.abs()

    if method == "density":
        support = _mk_support_by_density(Wabs, density)
    elif method == "absolute":
        support = _mk_support_by_abs_threshold(Wabs, thr)
    elif method == "topk":
        support = _mk_support_by_topk_rows(Wabs, k)
    else:
        raise ValueError("method must be one of {'density','absolute','topk'}")

    if share_support_by_abs:
        A_pos = torch.clamp(C, min=0) * support
        A_neg = torch.clamp(-C, min=0) * support
    else:
        A_pos = torch.clamp(C, min=0)
        A_neg = torch.clamp(-C, min=0)
        if method == "density":
            A_pos = A_pos * _mk_support_by_density(A_pos, density)
            A_neg = A_neg * _mk_support_by_density(A_neg, density)
    A_pos[torch.abs(A_pos) < 1e-12] = 0.0
    A_neg[torch.abs(A_neg) < 1e-12] = 0.0
    return A_pos, A_neg

def batch_corr_to_adj_np(
    corr_bnn: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    assert corr_bnn.ndim == 3
    A_list = [corr_to_adj_np(corr_bnn[b], **kwargs) for b in range(corr_bnn.shape[0])]
    return torch.stack(A_list, dim=0)

def batch_corr_to_signed_adjs_np(
    corr_bnn: torch.Tensor,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    B = corr_bnn.shape[0]
    pos_list, neg_list = [], []
    for b in range(B):
        Apos, Aneg = corr_to_signed_adjs_np(corr_bnn[b], **kwargs)
        pos_list.append(Apos)
        neg_list.append(Aneg)
    return torch.stack(pos_list, 0), torch.stack(neg_list, 0)

def node_features_from_adj_np(
    A: torch.Tensor,
    k_pe: int = 8,
    normalize_weight: bool = True,
    use_abs_for_metrics: bool = True,
    norm_mode: str = "dataset",            # "graph" | "dataset"
    norm_stats: dict | None = None       # {"mean": np.ndarray[C], "std": np.ndarray[C]}
) -> torch.Tensor:
    """
    返回 [N, C] 节点特征；当 norm_mode="dataset" 且给出 norm_stats 时，用数据集级 z-score。
    否则回退为图内 z-score。
    """
    A_np = A.detach().cpu().numpy().astype(np.float32, copy=False)
    N = A_np.shape[0]

    # === 计算 4 个拓扑统计 + k_pe 维 Laplacian PE ===
    W = np.abs(A_np) if use_abs_for_metrics else A_np.copy()
    np.fill_diagonal(W, 0.0)

    degree   = (W > 0).sum(axis=1, keepdims=True).astype(np.float32)
    strength = W.sum(axis=1, keepdims=True).astype(np.float32)

    Wn = W / (W.max() + 1e-12) if normalize_weight and W.max() > 0 else W
    clustering = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        nbrs = np.where(Wn[i] > 0)[0]
        k_i = len(nbrs)
        if k_i >= 2:
            s = 0.0
            for a in range(k_i):
                j = nbrs[a]
                for b in range(a + 1, k_i):
                    k = nbrs[b]
                    if Wn[j, k] > 0:
                        s += (Wn[i, j] * Wn[i, k] * Wn[j, k]) ** (1.0 / 3.0)
            clustering[i, 0] = (2.0 * s) / (k_i * (k_i - 1))

    local_eff = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        nbrs = np.where(W[i] > 0)[0]
        k_i = len(nbrs)
        if k_i >= 2:
            subW = W[np.ix_(nbrs, nbrs)]
            with np.errstate(divide='ignore'):
                subLen = np.where(subW > 0, 1.0 / (subW + 1e-12), 0.0)
            from scipy.sparse.csgraph import shortest_path
            spd = shortest_path(sp.csr_matrix(subLen), directed=False, unweighted=False)
            # 使用安全除法，避免对 0 或非有限值做除法，从而产生 RuntimeWarning
            mask = np.isfinite(spd) & (spd > 0)
            inv = np.zeros_like(spd, dtype=np.float64)
            if mask.any():
                np.divide(1.0, spd, out=inv, where=mask)
            # inv 中其余位置保持 0.0
            local_eff[i, 0] = (inv.sum() - np.sum(np.diag(inv))) / (k_i * (k_i - 1))

    D = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-12))
    L = np.eye(N, dtype=np.float32) - D_inv_sqrt @ W @ D_inv_sqrt
    evals, evecs = eigh(L)
    k_eff = min(k_pe, max(0, N - 1))
    pe = evecs[:, 1:1 + k_eff] if k_eff > 0 else np.zeros((N, 0), dtype=np.float32)

    X = np.concatenate([degree, strength, clustering, local_eff, pe.astype(np.float32, copy=False)], axis=1)  # [N,C]

    # === 规范化 ===
    if norm_mode == "dataset" and norm_stats is not None:
        mu  = norm_stats["mean"].reshape(1, -1).astype(np.float32, copy=False)
        std = norm_stats["std"].reshape(1, -1).astype(np.float32, copy=False)
        X = (X - mu) / (std + 1e-6)
    else:
        # 默认：图内 z-score
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)

    return torch.from_numpy(X).to(A.device).float()



def adj_np_to_pyg_data(
    A: torch.Tensor,
    x_np: torch.Tensor | None = None,
    undirected: bool = True,
    keep_self_loops: bool = False
) -> Data:
    A_sp = sp.csr_matrix(A.cpu().numpy())
    edge_index, edge_weight = from_scipy_sparse_matrix(A_sp)
    edge_weight = edge_weight.float()    
    if not keep_self_loops:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if undirected:
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='sum')
    x = x_np.float() if x_np is not None else torch.eye(A.shape[0], dtype=torch.float32, device=A.device)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

def corr_np_to_pyg_data(
    corr: torch.Tensor,
    sparsify_kwargs: dict | None = None,
    feat_kwargs: dict | None = None,
    **adj_kwargs
) -> Data:
    A = corr_to_adj_np(corr, **(sparsify_kwargs or {}))
    x_np = node_features_from_adj_np(A, **(feat_kwargs or {}))
    return adj_np_to_pyg_data(A, x_np=x_np, **adj_kwargs)

def batch_corr_np_to_data_list(
    corr_bnn: torch.Tensor,
    sparsify_kwargs: dict | None = None,
    feat_kwargs: dict | None = None,
    **adj_kwargs
) -> list[Data]:
    B = corr_bnn.shape[0]
    data_list = []
    for b in range(B):
        data_list.append(corr_np_to_pyg_data(corr_bnn[b], sparsify_kwargs, feat_kwargs, **adj_kwargs))
    return data_list

def signed_adj_np_to_pyg_data(
    A_pos: torch.Tensor,
    A_neg: torch.Tensor,
    x_np: torch.Tensor | None = None,
    undirected: bool = True,
    keep_self_loops: bool = False
) -> Data:
    pos_ei, pos_ew = from_scipy_sparse_matrix(sp.csr_matrix(A_pos.cpu().numpy()))
    neg_ei, neg_ew = from_scipy_sparse_matrix(sp.csr_matrix(A_neg.cpu().numpy()))

    if not keep_self_loops:
        pos_ei, pos_ew = remove_self_loops(pos_ei, pos_ew)
        neg_ei, neg_ew = remove_self_loops(neg_ei, neg_ew)
    if undirected:
        pos_ei, pos_ew = to_undirected(pos_ei, pos_ew, reduce='sum')
        neg_ei, neg_ew = to_undirected(neg_ei, neg_ew, reduce='sum')

    x = x_np.float() if x_np is not None else torch.eye(A_pos.shape[0], dtype=torch.float32, device=A_pos.device)
    data = Data(x=x)
    data.pos_edge_index, data.pos_edge_attr = pos_ei, pos_ew
    data.neg_edge_index, data.neg_edge_attr = neg_ei, neg_ew
    return data

def signed_corr_np_to_pyg_data(
    corr: torch.Tensor,
    sparsify_kwargs: dict | None = None,
    feat_kwargs: dict | None = None,
    **adj_kwargs
) -> Data:
    A_pos, A_neg = corr_to_signed_adjs_np(corr, **(sparsify_kwargs or {}))
    A_mag = A_pos + A_neg
    x_np = node_features_from_adj_np(A_mag, **(feat_kwargs or {}))
    return signed_adj_np_to_pyg_data(A_pos, A_neg, x_np=x_np, **adj_kwargs)

def batch_signed_corr_np_to_data_list(
    corr_bnn: torch.Tensor,
    sparsify_kwargs: dict | None = None,
    feat_kwargs: dict | None = None,
    **adj_kwargs
) -> list[Data]:
    B = corr_bnn.shape[0]
    data_list = []
    for b in range(B):
        data_list.append(signed_corr_np_to_pyg_data(corr_bnn[b], sparsify_kwargs, feat_kwargs, **adj_kwargs))
    return data_list

def data_list_to_batch(data_list: list[Data]) -> Batch:
    return Batch.from_data_list(data_list)

def build_mask_vec_from_other_adj_bnn(
    other_adj_bnn: torch.Tensor,
    target_batch: Batch,
    select_top_nodes
) -> torch.Tensor:
    assert other_adj_bnn.ndim == 3
    bool_masks = select_top_nodes(other_adj_bnn)
    assert bool_masks.ndim == 2
    ptr = target_batch.ptr
    B = ptr.numel() - 1
    assert B == bool_masks.shape[0], "Batch 样本数与 mask 批量不一致"

    parts = []
    for i in range(B):
        n_i = int(ptr[i + 1] - ptr[i])
        mask_i = bool_masks[i]
        assert mask_i.shape[0] == n_i, f"第 {i} 个图节点数 {n_i} 与 mask 长度 {mask_i.shape[0]} 不一致"
        parts.append(mask_i.float())
    mask_vec = torch.cat(parts, dim=0).to(target_batch.x.device)
    return mask_vec

def dense_adj_torch_to_edge_index(adj_t: torch.Tensor,
                                  threshold: float = 0.0,
                                  undirected: bool = True,
                                  keep_self_loops: bool = False):
    if threshold > 0:
        adj_t = adj_t.clone()
        adj_t[adj_t <= threshold] = 0
    edge_index, edge_weight = dense_to_sparse(adj_t)
    if not keep_self_loops:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if undirected:
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='sum')
    return edge_index, edge_weight

def init_graph_batch_from_corr(
    corr_bnn: torch.Tensor,
    select_top_nodes,
    sparsify_kwargs: dict = None,
    feat_kwargs: dict = None,
    adj_kwargs: dict = None,
) -> dict:
    sparsify_kwargs = sparsify_kwargs or {}
    feat_kwargs = feat_kwargs or {}
    adj_kwargs = adj_kwargs or {}

    adj_bnn = batch_corr_to_adj_np(corr_bnn, **sparsify_kwargs)
    data_list = batch_corr_np_to_data_list(
        corr_bnn, sparsify_kwargs=sparsify_kwargs, feat_kwargs=feat_kwargs, **adj_kwargs
    )
    batch = data_list_to_batch(data_list)
    mask = build_mask_vec_from_other_adj_bnn(adj_bnn, batch, select_top_nodes)
    return {
        'batch': batch,
        'mask': mask,
        'adj_bnn': adj_bnn,
        'data_list': data_list,
    }


