from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def inverse_transform_coords(
    pred_norm: np.ndarray,
    scaler: StandardScaler,
    feature_dim: int = 4,
) -> np.ndarray:
    N, T, D = pred_norm.shape
    pred_flat = pred_norm.reshape(-1, D)            # (N*T, D)
    pred_phys = scaler.inverse_transform(pred_flat)  # (N*T, D)
    pred_phys = pred_phys.reshape(N, T, D)           # (N, T, D)
    return pred_phys

def compute_attention_summary(attn_weights: torch.Tensor) -> dict:
    # 对所有 batch 和 head 取平均 → (seq_len, seq_len)
    avg_attn = attn_weights.mean(dim=(0, 1)).detach().cpu().numpy()

    # 每个 head 的平均注意力权重 → (nhead,)
    head_importance = attn_weights.mean(dim=(0, 2, 3)).detach().cpu().numpy()

    # 每个 query 位置对所有 key 的平均注意力 → (seq_len,)
    temporal_bias = avg_attn.mean(axis=1)

    # 对最后 5 个 key 位置的注意力（即对"最近"历史的关注程度）
    seq_len = avg_attn.shape[0]
    recent_bias = avg_attn[:, -5:].sum() / avg_attn.sum()

    return {
        "avg_attention": avg_attn,
        "head_importance": head_importance,
        "temporal_bias": temporal_bias,
        "recent_bias_ratio": float(recent_bias),
    }

def compute_ade_fde(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    
    # 只取位置 (x, y)，忽略速度
    # (N, future_len, 4) → (N, future_len, 2)
    pred_pos = y_pred[:, :, :2]
    true_pos = y_true[:, :, :2]

    # 逐帧欧氏距离: ||(dx, dy)||₂
    # (N, future_len, 2) → (N, future_len)
    errors = np.linalg.norm(pred_pos - true_pos, axis=-1)

    # ADE: 全部帧的平均
    ade = float(errors.mean())

    # FDE: 只看最后一帧
    fde = float(errors[:, -1].mean())

    # 每帧的平均误差 (用于画 "误差 vs 预测时域" 曲线)
    errors_per_step = errors.mean(axis=0)  # (future_len,)

    return ade, fde, errors_per_step