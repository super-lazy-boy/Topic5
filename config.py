import torch
import numpy as np
CONFIG = {
    # ── 数据路径 ──
    "data_path": r"D:\ZAQ\code\LP\Topic5\data",

    # ── 数据加载 ──
    "num_recordings": 10,       # 加载前 N 个录像 (1~60, 越多数据效果越好但越慢)
    "min_track_len": 60,       # 最短轨迹长度 ( < history+future 的轨迹会被丢弃)

    # ── 序列参数 ──
    "history_len": 20,         # 历史(观测)帧数 → 20 帧 = 0.8 秒 (highD 是 25Hz)
    "future_len": 30,          # 未来(预测)帧数 → 30 帧 = 1.2 秒

    # ── MTR 预处理 ──
    "use_agent_centric": True, # 是否启用目标车坐标系变换 (MTR 核心, 强烈推荐)
    "use_interaction": True,   # 是否提取周围车辆交互特征 (已接入 Decoder 模型)

    # ── 模型超参数 ──
    # "input_dim": 28,           # 输入特征维度: 4(基础: x,y,vx,vy) + 24(交互: 6方向×4特征)
    "d_model": 64,             # Transformer 隐空间维度
    "nhead": 4,                # 多头注意力头数
    "num_layers": 2,           # Transformer Encoder 层数
    "dim_feedforward": 128,    # FFN 中间层维度
    "dropout": 0.1,            # Dropout 概率

    # ── 训练超参数 ──
    "batch_size": 16,          # 批大小 (CPU 训练建议 32~128)
    "epochs": 30,              # 最大训练轮数
    "lr": 0.001,               # 初始学习率
    "weight_decay": 1e-5,      # L2 正则化系数
    "lr_step_size": 10,        # 学习率衰减步长 (每 N 个 epoch 衰减一次)
    "lr_gamma": 0.5,           # 学习率衰减因子
    "patience": 8,             # 早停耐心值 (验证 loss 不降 N 轮后停止)

    # ── 数据划分 ──
    "train_ratio": 0.70,       # 训练集比例
    "val_ratio": 0.15,         # 验证集比例 (测试集 = 1 - train - val)

    # ── 设备 ──
    "device": "cuda" if torch.cuda.is_available() else "cpu",          

    # ── 随机种子 ──
    "seed": 42,
}