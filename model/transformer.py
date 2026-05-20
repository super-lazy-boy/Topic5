import torch
import torch.nn as nn
from model.encoder import PositionalEncoding


class TransformerTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,           # 输入特征维度 (x, y, z, vx, vy, vz, ax, ay, az, yaw, yaw_rate)
        output_dim: int = 4,           # 输出特征维度
        d_model: int = 64,             # 隐空间维度
        nhead: int = 4,                # 多头注意力的头数 (必须整除 d_model)
        num_layers: int = 2,           # Transformer Encoder 层数
        dim_feedforward: int = 128,    # FFN 中间层维度
        future_len: int = 30,          # 预测的未来帧数
        dropout: float = 0.1,          # Dropout 概率
        num_decoder_layers: int = 1,   # Transformer Decoder 层数 (新增参数, 默认 1)
    ):
        super().__init__()
        self.d_model = d_model
        self.future_len = future_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model)

        # Shape: (B, 20, 64) → (B, 20, 64)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=500, dropout=dropout
        )

        # Shape: (B, 20, 64) → (B, 20, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.query_embed = nn.Parameter(
            torch.randn(future_len, d_model)
        )
        # Shape: (future_len, d_model) = (30, 64)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        # Shape: (B, 30, 64) → (B, 30, 4)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = self.pos_encoder(x)

        if return_attention:
            # 手动逐层执行，以捕获 Encoder 自注意力权重用于可视化
            src = x
            attn_weights_list = []

            for layer in self.transformer_encoder.layers:
                # 自注意力: Q=K=V=src，每个历史帧关注所有历史帧
                attn_out, attn_weights = layer.self_attn(
                    src, src, src,
                    need_weights=True,
                    average_attn_weights=False,
                )
                # attn_out:      (B, 20, 64)
                # attn_weights:  (B, nhead, 20, 20) ← 每头一个 20×20 注意力矩阵

                attn_weights_list.append(attn_weights)

                # Add & Norm + Feed-Forward
                src = layer.norm1(src + layer.dropout1(attn_out))
                ff_out = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(src)))
                )
                src = layer.norm2(src + layer.dropout2(ff_out))

            encoded = src  # (B, 20, 64) — Memory (作为 K, V)
            # 取最后一层的自注意力权重返回
            attn_weights = attn_weights_list[-1]  # (B, nhead, 20, 20)
        else:
            encoded = self.transformer_encoder(x)
            # Shape: (B, 20, 64)
            attn_weights = None

        query = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        # query: (B, 30, 64) — 作为 Decoder 的 Target (Query)

        decoded = self.transformer_decoder(
            tgt=query,    # (B, 30, 64) — Target/Query
            memory=encoded,  # (B, 20, 64) — Key & Value
        )
        # decoded: (B, 30, 64)

        # (B, 30, 64) → (B, 30, output_dim)
        out = self.output_proj(decoded)

        if return_attention:
            return out, attn_weights
        return out


