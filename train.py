import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim

# 导入本项目的模块
from datasets.dataset import create_dataloaders

from model.transformer import  TransformerTrajectoryPredictor
from model.encoder import PositionalEncoding
from utils import inverse_transform_coords, compute_attention_summary, compute_ade_fde
from datasets.data_utils import load_multiple_recordings, load_recording, create_sequences

from config import CONFIG

# 设置随机种子以保证可复现性
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
device = torch.device(CONFIG["device"])




def load_and_prepare_data(config: dict):

    data_path = config["data_path"]
    num_rec = config["num_recordings"]

    tracks_df, recordingmeta_df = load_multiple_recordings(data_path, list(range(1, num_rec + 1)))

    X, y, scaler, track_lengths = create_sequences(
        tracks_df, 
        history_len=config["history_len"],
        future_len=config["future_len"],
        min_track_len=config["min_track_len"],
        use_agent_centric=config["use_agent_centric"]
    )

    # selected_meta = recordingmeta_df[recordingmeta_df["id"].isin(range(1, num_rec + 1))] 
    # total_vehicles = selected_meta["numVehicles"].sum()


    train_loader, val_loader, test_loader,X_train, X_val, X_test,y_train, y_val, y_test = create_dataloaders(X, y, track_lengths=track_lengths,
                                                  batch_size=config["batch_size"],
                                                    train_ratio=config["train_ratio"],
                                                    val_ratio=config["val_ratio"],)
    input_dim = X_train.shape[-1]

    # 保留测试集的物理坐标版本 (用于可视化)
    X_test_phys = inverse_transform_coords(X_test, scaler)
    y_test_phys = inverse_transform_coords(y_test, scaler)

    print(f"\n  数据加载完成！")
    print(f"  训练集: {len(train_loader):,} 样本")
    print(f"  验证集: {len(val_loader):,}   样本")
    print(f"  测试集: {len(test_loader):,} 样本")

    return train_loader, val_loader, test_loader, scaler, X_test_phys, y_test_phys,input_dim


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            pred = model(X_batch)
            target_pos = y_batch[:, :, 0:2]
            loss = criterion(pred, target_pos)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()  # 禁用梯度计算 → 节省内存和计算
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:

    model.eval()  # 切换到评估模式 (关闭 Dropout)
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            pred = model(X_batch)
            target_pos = y_batch[:, :, 0:2]
            loss = criterion(pred, target_pos)
        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> dict:
    device = torch.device(config["device"])
    model = model.to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    # ── 损失函数与优化器 ──
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.1,
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"  开始训练 (max {config['epochs']} epochs)")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5s}  {'Train Loss':>12s}  {'Val Loss':>12s}  "
          f"{'LR':>10s}  {'Status':>12s}")
    print(f"  {'-'*58}")

    for epoch in range(1, config["epochs"] + 1):
        # ── 训练 ──
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # ── 验证 ──
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # ── 记录 ──
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ── 学习率衰减 ──
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── 早停检查 ──
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            status = "✓ best"
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  {'-'*58}")
                print(f"  早停触发! 最佳 epoch: {best_epoch}, "
                      f"最佳 val loss: {best_val_loss:.6f}")
                break

        # 每 5 个 epoch 或第 1 个 epoch 时输出
        if epoch % 5 == 0 or epoch == 1 or status == "✓ best":
            print(f"  {epoch:5d}  {train_loss:12.6f}  {val_loss:12.6f}  "
                  f"{current_lr:10.2e}  {status:>12s}")

    print(f"\n  训练完成! "
          f"(最佳 epoch: {best_epoch}, 最佳 val loss: {best_val_loss:.6f})")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler,
    config: dict,
) -> dict:

    device = torch.device(config["device"])
    model.eval()

    all_preds, all_targets = [], []

    print(f"\n[评估] 在测试集上推理...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)                    # (B, 30, 4)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    # 拼接所有 batch → (N_test, 30, 4)
    y_pred_norm = np.concatenate(all_preds, axis=0)
    y_true_norm = np.concatenate(all_targets, axis=0)

    y_true_norm = y_true_norm[:, :, 0:2]

    mean_p = scaler.mean_[0:2]
    scale_p = scaler.scale_[0:2]

    y_pred_phys = y_pred_norm * scale_p + mean_p
    y_true_phys = y_true_norm * scale_p + mean_p

    # 计算指标
    ade, fde, errors_per_step = compute_ade_fde(y_pred_phys, y_true_phys)

    return {
        "ade": ade,
        "fde": fde,
        "errors_per_step": errors_per_step,
        "y_pred_phys": y_pred_phys,
        "y_true_phys": y_true_phys,
    }


def visualize_training_curves(history: dict):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图: Loss 曲线 ──
    ax = axes[0]
    epochs = range(1, len(history["train_losses"]) + 1)
    ax.plot(epochs, history["train_losses"], "b-", linewidth=2, label="训练 Loss")
    ax.plot(epochs, history["val_losses"], "r-", linewidth=2, label="验证 Loss")
    ax.axvline(x=history["best_epoch"], color="gray", linestyle="--", alpha=0.7,
               label=f"最佳 epoch ({history['best_epoch']})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Transformer 训练曲线", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── 右图: 误差 vs 预测时域 ──
    # 此图在 evaluate_model 之后填充, 先创建空白轴
    ax = axes[1]
    ax.set_xlabel("预测时域 (帧 @ 25Hz)", fontsize=12)
    ax.set_ylabel("位移误差 (m)", fontsize=12)
    ax.set_title("预测误差随时间的增长", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def visualize_predictions(
    X_test_phys: np.ndarray,
    y_true_phys: np.ndarray,
    y_pred_phys: np.ndarray,
    num_samples: int = 5,
    history_len: int = 20,
):

    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]

        # 历史轨迹 (黑色)
        hist = X_test_phys[i]
        ax.plot(hist[:, 0], hist[:, 1], "k.-", linewidth=2, markersize=3,
                label="历史轨迹 (20帧)")

        # 真实未来 (蓝色)
        gt = y_true_phys[i]
        ax.plot(gt[:, 0], gt[:, 1], "b.-", linewidth=2, markersize=3,
                label="真实未来 (30帧)")

        # 预测未来 (红色)
        pred = y_pred_phys[i]
        ax.plot(pred[:, 0], pred[:, 1], "r.--", linewidth=2, markersize=3,
                label="预测未来 (30帧)")

        # 标记起点和终点
        # 历史最后一帧 → 预测起点 (当前位置)
        ax.scatter(hist[-1, 0], hist[-1, 1], c="k", s=80, zorder=5,
                   marker="o", label="当前时刻")
        # 预测的最终位置
        ax.scatter(pred[-1, 0], pred[-1, 1], c="r", s=80, zorder=5,
                   marker="x", label="预测终点")

        # 计算该样本的 FDE
        sample_fde = np.linalg.norm(pred[-1, :2] - gt[-1, :2])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"样本 {i+1}  (FDE = {sample_fde:.2f} m)")
        ax.legend(fontsize=7, loc="best")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("轨迹预测结果对比 (Transformer)", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def visualize_attention(
    model: nn.Module,
    X_test: np.ndarray,
    device: torch.device,
    num_samples: int = 1,
):

    model.eval()

    # 取前 num_samples 个样本
    sample_X = torch.from_numpy(X_test[:num_samples]).float().to(device)
    # Shape: (num_samples, 20, 4)

    with torch.no_grad():
        _, attn_weights = model(sample_X, return_attention=True)
    # attn_weights: (num_samples, nhead, seq_len, seq_len)
    # 例如: (1, 4, 20, 20)

    # 对所有头和样本求平均 → (seq_len, seq_len) = (20, 20)
    weights_avg = attn_weights.mean(dim=(0, 1)).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图 1: 平均注意力热力图 ──
    ax = axes[0]
    im = ax.imshow(weights_avg, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xlabel("Key (被关注的时间步)", fontsize=11)
    ax.set_ylabel("Query (当前时间步)", fontsize=11)
    ax.set_title("自注意力矩阵 (所有头的平均)", fontsize=13)
    # 标注时间标签
    tick_positions = [0, 5, 10, 15, 19]
    tick_labels = ["t=0\n(1.2s前)", "t=5", "t=10", "t=15", "t=19\n(当前)"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="注意力权重", shrink=0.8)

    # ── 子图 2: 各头的重要性 ──
    ax = axes[1]
    # 每个头在所有 batch 上的平均注意力
    head_weights = attn_weights.mean(dim=(0, 2, 3)).cpu().numpy()  # (nhead,)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(head_weights)))
    bars = ax.bar(range(len(head_weights)), head_weights, color=colors, edgecolor="gray")
    ax.set_xlabel("注意力头编号", fontsize=11)
    ax.set_ylabel("平均注意力权重", fontsize=11)
    ax.set_title("各注意力头的重要性", fontsize=13)
    ax.set_xticks(range(len(head_weights)))
    ax.set_xticklabels([f"头 {i}" for i in range(len(head_weights))])
    # 在柱上标数值
    for bar, val in zip(bars, head_weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", fontsize=9)

    # ── 子图 3: 时间偏向 — 注意力是否偏向近期？ ──
    ax = axes[2]
    seq_len = weights_avg.shape[0]
    # 每个 query 位置的平均 attention
    avg_by_query = weights_avg.mean(axis=1)  # (20,) — 每帧"接收"多少关注
    ax.plot(range(seq_len), avg_by_query, "o-", color="steelblue",
            linewidth=2, markersize=8)
    # 标注"近期"和"远期"
    ax.axvspan(15, 19, alpha=0.15, color="orange", label="近期 (t=15~19)")
    ax.axvspan(0, 5, alpha=0.1, color="green", label="远期 (t=0~5)")
    ax.set_xlabel("时间步位置", fontsize=11)
    ax.set_ylabel("平均注意力权重", fontsize=11)
    ax.set_title("注意力的时间分布", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Transformer 自注意力权重可视化 — 模型在\"看\"哪里？",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    return fig


def main():
    
    train_loader, val_loader, test_loader, scaler, X_test_phys, y_test_phys, input_dim = load_and_prepare_data(CONFIG)

    model = TransformerTrajectoryPredictor(
        input_dim=input_dim,
        output_dim=CONFIG["output_dim"],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        future_len=CONFIG["future_len"],
        dropout=CONFIG["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())

    history = train_model(model, train_loader, val_loader, CONFIG)

    eval_results = evaluate_model(model, test_loader, scaler, CONFIG)


    fig1, (ax_loss, ax_error) = visualize_training_curves(history)
    future_len = CONFIG["future_len"]
    ax_error.plot(range(1, future_len + 1), eval_results["errors_per_step"],
                  "o-", color="steelblue", markersize=3, linewidth=2)
    ax_error.fill_between(range(1, future_len + 1),
                          eval_results["errors_per_step"], alpha=0.2,
                          color="steelblue")
    ax_error.axhline(y=eval_results["ade"], color="red", linestyle="--",
                     alpha=0.6, label=f"ADE = {eval_results['ade']:.3f} m")
    ax_error.legend(fontsize=10)
    plt.tight_layout()
    num_viz = min(5, len(X_test_phys))

    fig2 = visualize_predictions(
        X_test_phys,
        eval_results["y_true_phys"],
        eval_results["y_pred_phys"],
        num_samples=num_viz,
        history_len=CONFIG["history_len"],
    )

    # 从 test_loader 取一些样本
    test_samples_X = []
    for Xb, _ in test_loader:
        test_samples_X.append(Xb.numpy())
        if len(np.concatenate(test_samples_X, axis=0)) >= 8:
            break
    X_attn = np.concatenate(test_samples_X, axis=0)[:8]

    fig3 = visualize_attention(model, X_attn, device)
    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(str(save_dir / "01_training_history.png"), dpi=300, bbox_inches='tight')
    fig2.savefig(str(save_dir / "02_trajectory_predictions.png"), dpi=300, bbox_inches='tight')
    fig3.savefig(str(save_dir / "03_attention_heatmap.png"), dpi=300, bbox_inches='tight')

    # 显示所有图形
    plt.show()

    return model, history, eval_results


if __name__ == "__main__":
    model, history, eval_results = main()
