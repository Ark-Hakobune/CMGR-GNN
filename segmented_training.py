"""
分段训练模块：为不同年龄段的数据分别训练预测头
支持多任务学习框架，共享编码器但使用段特定的预测头
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class SegmentedAgePredictor(nn.Module):
    """
    为不同年龄段维护独立的预测头，共享主编码器。
    
    策略：在双模态融合后提取共享表示，然后为每个段使用不同的预测头。
    
    参数：
    - encoder: 共享的双模态编码器
    - fusion: 基础融合模块（用于提取共享表示）
    - segment_edges: 年龄段边界列表，例如 [45, 135, 225, 315, 450, 630, 810]
    - embed_dim: 编码器输出维度
    - hidden_dim: 预测头隐层维度
    - dropout: Dropout比率
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        fusion: nn.Module,
        segment_edges: List[float] = None,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if segment_edges is None:
            segment_edges = [45, 135, 225, 315, 450, 630, 810]
        
        self.encoder = encoder
        self.fusion_module = fusion
        self.segment_edges = segment_edges
        self.num_segments = len(segment_edges) - 1
        self.embed_dim = embed_dim
        
        # 为每个年龄段创建独立的预测头
        # 注意：fusion 输出标量，所以我们在 fusion 之前提取特征
        # 策略：使用 fusion 的中间层输出作为特征，或者直接从编码器输出提取
        # 这里我们使用编码器输出的平均池化
        self.segment_heads = nn.ModuleList()
        for i in range(self.num_segments):
            head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.segment_heads.append(head)
        
        # 全局预测头（用于所有数据的联合训练）
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def get_segment_idx(self, ages: torch.Tensor) -> torch.Tensor:
        """
        根据年龄返回对应的段索引。
        
        返回：
        - segment_idx: (B,) 张量，每个样本对应的段索引
        """
        B = ages.size(0)
        segment_idx = torch.zeros(B, dtype=torch.long, device=ages.device)
        
        for i in range(self.num_segments):
            if i == 0:
                mask = (ages >= self.segment_edges[i]) & (ages <= self.segment_edges[i+1])
            else:
                mask = (ages > self.segment_edges[i]) & (ages <= self.segment_edges[i+1])
            
            segment_idx[mask] = i
        
        return segment_idx
    
    def forward(
        self,
        func_batch,
        morph_batch,
        ages: Optional[torch.Tensor] = None,
        use_segment_heads: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。
        
        参数：
        - func_batch, morph_batch: 图数据批次
        - ages: 样本年龄（如果指定则使用段特定的头）
        - use_segment_heads: 是否使用段特定的预测头
        
        返回：
        - y_pred: 预测值 (B, 1) 或 (B,)
        - segment_idx: 每个样本的段索引（可选）
        """
        # 共享编码
        zf, zm = self.encoder(func_batch, morph_batch)
        
        # 提取特征：这里使用融合模块的中间结果
        # 为了获得合适维度的特征向量，我们对两个模态的编码进行池化
        # 假设 zf, zm 是全局池化后的向量或序列
        if zf.dim() == 3:  # [B, L, D] - 序列
            shared_embedding = torch.cat([zf.mean(dim=1), zm.mean(dim=1)], dim=-1)  # [B, 2*D]
        else:  # [B, D] - 已经池化
            shared_embedding = torch.cat([zf, zm], dim=-1)  # [B, 2*D]
        
        # 如果共享表示维度大于 embed_dim，做一个投影
        if shared_embedding.size(-1) != self.embed_dim:
            # 创建动态投影层
            if not hasattr(self, '_projection'):
                self._projection = nn.Linear(shared_embedding.size(-1), self.embed_dim).to(shared_embedding.device)
            shared_embedding = self._projection(shared_embedding)
        
        # 如果不使用段特定头或未提供年龄，使用全局头
        if not use_segment_heads or ages is None:
            y_pred = self.global_head(shared_embedding)
            return y_pred.squeeze(-1) if y_pred.size(-1) == 1 else y_pred, None
        
        # 使用段特定的预测头
        B = shared_embedding.size(0)
        y_pred = torch.zeros(B, device=shared_embedding.device)
        segment_idx = self.get_segment_idx(ages)
        
        for seg_id in range(self.num_segments):
            mask = (segment_idx == seg_id)
            if mask.sum() > 0:
                seg_embedding = shared_embedding[mask]
                seg_pred = self.segment_heads[seg_id](seg_embedding).squeeze(-1)
                y_pred[mask] = seg_pred
        
        return y_pred, segment_idx
    
    def forward_with_head_selection(
        self,
        func_batch,
        morph_batch,
        ages: torch.Tensor,
        head_type: str = "segment"
    ) -> torch.Tensor:
        """
        灵活的前向传播，支持选择不同的预测头。
        
        参数：
        - head_type: "segment" (段特定), "global" (全局), "ensemble" (平均)
        
        返回：
        - y_pred: 预测值
        """
        zf, zm = self.encoder(func_batch, morph_batch)
        
        # 提取共享表示
        if zf.dim() == 3:  # [B, L, D]
            shared_embedding = torch.cat([zf.mean(dim=1), zm.mean(dim=1)], dim=-1)
        else:  # [B, D]
            shared_embedding = torch.cat([zf, zm], dim=-1)
        
        if shared_embedding.size(-1) != self.embed_dim:
            if not hasattr(self, '_projection'):
                self._projection = nn.Linear(shared_embedding.size(-1), self.embed_dim).to(shared_embedding.device)
            shared_embedding = self._projection(shared_embedding)
        
        if head_type == "segment":
            B = shared_embedding.size(0)
            y_pred = torch.zeros(B, device=shared_embedding.device)
            segment_idx = self.get_segment_idx(ages)
            
            for seg_id in range(self.num_segments):
                mask = (segment_idx == seg_id)
                if mask.sum() > 0:
                    seg_embedding = shared_embedding[mask]
                    seg_pred = self.segment_heads[seg_id](seg_embedding).squeeze(-1)
                    y_pred[mask] = seg_pred
            
            return y_pred
        
        elif head_type == "global":
            return self.global_head(shared_embedding).squeeze(-1)
        
        elif head_type == "ensemble":
            # 将所有段头的预测平均
            segment_idx = self.get_segment_idx(ages)
            predictions = []
            
            for seg_id in range(self.num_segments):
                mask = (segment_idx == seg_id)
                if mask.sum() > 0:
                    seg_embedding = shared_embedding[mask]
                    seg_pred = self.segment_heads[seg_id](seg_embedding).squeeze(-1)
                    predictions.append((mask, seg_pred))
            
            y_pred = torch.zeros_like(ages)
            for mask, seg_pred in predictions:
                y_pred[mask] = seg_pred
            
            return y_pred
        
        else:
            raise ValueError(f"Unknown head_type: {head_type}")


class SegmentedLoss(nn.Module):
    """
    分段损失函数：可以为每个段赋予不同的权重或使用不同的损失函数。
    """
    
    def __init__(
        self,
        segment_weights: Optional[Dict[int, float]] = None,
        segment_edges: List[float] = None,
        base_loss_fn = None,
    ):
        super().__init__()
        
        if segment_edges is None:
            segment_edges = [45, 135, 225, 315, 450, 630, 810]
        
        self.segment_edges = segment_edges
        self.num_segments = len(segment_edges) - 1
        
        # 默认使用 SmoothL1Loss
        if base_loss_fn is None:
            base_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.base_loss_fn = base_loss_fn
        
        # 段权重（如果为None则所有段权重相同）
        if segment_weights is None:
            segment_weights = {i: 1.0 for i in range(self.num_segments)}
        self.segment_weights = segment_weights
    
    def get_segment_idx(self, ages: torch.Tensor) -> torch.Tensor:
        """根据年龄返回段索引"""
        B = ages.size(0)
        segment_idx = torch.zeros(B, dtype=torch.long, device=ages.device)
        
        for i in range(self.num_segments):
            if i == 0:
                mask = (ages >= self.segment_edges[i]) & (ages <= self.segment_edges[i+1])
            else:
                mask = (ages > self.segment_edges[i]) & (ages <= self.segment_edges[i+1])
            
            segment_idx[mask] = i
        
        return segment_idx
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        segment_idx: Optional[torch.Tensor] = None,
        return_by_segment: bool = False,
    ) -> torch.Tensor | Dict[int, torch.Tensor]:
        """
        计算分段损失。
        
        参数：
        - y_pred: 预测值 (B,)
        - y_true: 真实值 (B,)
        - segment_idx: 段索引（如果为None则根据y_true计算）
        - return_by_segment: 是否返回按段分解的损失
        
        返回：
        - 如果 return_by_segment=False: 加权平均损失（标量）
        - 如果 return_by_segment=True: 字典 {seg_id: loss}
        """
        if segment_idx is None:
            segment_idx = self.get_segment_idx(y_true)
        
        per_sample_loss = self.base_loss_fn(y_pred, y_true)
        
        if return_by_segment:
            segment_losses = {}
            for seg_id in range(self.num_segments):
                mask = (segment_idx == seg_id)
                if mask.sum() > 0:
                    seg_loss = per_sample_loss[mask].mean()
                    segment_losses[seg_id] = seg_loss.item()
                else:
                    segment_losses[seg_id] = 0.0
            
            return segment_losses
        
        else:
            # 应用段权重
            weighted_loss = torch.zeros_like(per_sample_loss)
            for seg_id in range(self.num_segments):
                mask = (segment_idx == seg_id)
                weight = self.segment_weights.get(seg_id, 1.0)
                weighted_loss[mask] = per_sample_loss[mask] * weight
            
            return weighted_loss.mean()


def create_segment_data_splits(
    all_ages: np.ndarray,
    segment_edges: List[float] = None,
) -> Dict[int, np.ndarray]:
    """
    将数据按年龄段分割。
    
    返回：
    - 字典 {segment_id: 样本索引数组}
    """
    if segment_edges is None:
        segment_edges = [45, 135, 225, 315, 450, 630, 810]
    
    segment_splits = {}
    num_segments = len(segment_edges) - 1
    
    for seg_id in range(num_segments):
        if seg_id == 0:
            mask = (all_ages >= segment_edges[seg_id]) & (all_ages <= segment_edges[seg_id+1])
        else:
            mask = (all_ages > segment_edges[seg_id]) & (all_ages <= segment_edges[seg_id+1])
        
        indices = np.where(mask)[0]
        segment_splits[seg_id] = indices
    
    return segment_splits
