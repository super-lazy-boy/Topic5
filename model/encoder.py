import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 64, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建 (max_len, d_model) 的零矩阵，后面填入 sin/cos 值
        pe = torch.zeros(max_len, d_model)  # Shape: (500, 64)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，变成 (1, max_len, d_model)，方便后续广播
        pe = pe.unsqueeze(0)  # (500, 64) → (1, 500, 64)

        # register_buffer: 将 pe 注册为模型的持久状态
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
