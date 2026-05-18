import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from datasets.data_utils import load_recording, create_sequences
from utils import inverse_transform_coords

class TrajectoryDataset(Dataset):
    """
    【PyTorch Dataset】封装 MTR 预处理后的轨迹数据（支持 lazy subset）。

    当 X/y 为 np.memmap 时，torch.from_numpy 创建的 tensor 与 memmap
    共享内存，OS 按需换页，不会一次性加载全部数据到 RAM。

    === __getitem__ 返回的 Shape ===

      X_sample: (history_len, feature_dim)  例: (20, 28)
      y_sample: (future_len,  feature_dim)  例: (30, 28)

    参数:
        X:       NumPy 数组或 memmap, Shape = (N, history_len, feature_dim)
        y:       NumPy 数组或 memmap, Shape = (N, future_len,  feature_dim)
        indices: 可选，子集索引。为 None 则使用全部数据。
                 传入后 __getitem__ 会映射到对应样本，避免 X[indices] 复制。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray = None):
        # memmap → tensor 共享内存，OS 按需加载 → 内存高效
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.indices is not None:
            idx = self.indices[idx]
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集/验证集/测试集并创建 DataLoader。

    === 内存优化 ===
    使用 TrajectoryDataset 的 indices 参数实现 lazy subset：
    - train/val Dataset 持有完整 memmap + 子集索引，按需从磁盘读取
    - 不再做 X[train_idx] 的 fancy-index 拷贝（会加载全量到 RAM）
    - 仅 test 集做拷贝（15% 数据，约 3 GiB），供 train.py 做物理坐标还原
    """
    total = len(X)
    indices = np.random.permutation(total)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # ── Lazy Dataset（不拷贝数据，通过 indices 映射）──
    train_dataset = TrajectoryDataset(X, y, train_idx)
    val_dataset = TrajectoryDataset(X, y, val_idx)
    test_dataset = TrajectoryDataset(X, y, test_idx)

    # ── Test 集加载到内存（供 inverse_transform_coords 等可视化用途）──
    # 只有 15% 数据，约 3 GiB，可接受
    X_test = np.array(X[test_idx])
    y_test = np.array(y[test_idx])
    # train/val 返回 memmap 原始引用（train.py 不使用它们做直接计算）
    X_train, y_train = X, y
    X_val, y_val = X, y

    print(f"[create_dataloaders] 数据划分:")
    print(f"  训练集: {n_train:,} 样本 ({train_ratio*100:.0f}%) — lazy memmap")
    print(f"  验证集: {n_val:,}   样本 ({val_ratio*100:.0f}%) — lazy memmap")
    print(f"  测试集: {len(X_test):,} 样本 ({(1-train_ratio-val_ratio)*100:.0f}%) — 已加载")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    print(f"  DataLoader 创建完毕 (batch_size={batch_size})")
    print(f"  每个 batch: (B, {X.shape[1]}, {X.shape[2]})")

    return (train_loader, val_loader, test_loader,
            X_train, X_val, X_test,
            y_train, y_val, y_test)
