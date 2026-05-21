import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from datasets.data_utils import load_recording, create_sequences
from utils import inverse_transform_coords
import os
from typing import List, Dict, Any

class TrajectoryDataset(Dataset):

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


def create_dataloaders(X: np.ndarray,
                        y: np.ndarray,
                        track_lengths: List[int],  
                        batch_size: int = 64,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    num_vehicles = len(track_lengths)
    
    # ── Step 1: 按照车辆数量进行打乱划分 ──
    vehicle_indices = np.random.permutation(num_vehicles)

    n_train_veh = int(num_vehicles * train_ratio)
    n_val_veh = int(num_vehicles * val_ratio)

    train_veh_idx = vehicle_indices[:n_train_veh]
    val_veh_idx = vehicle_indices[n_train_veh : n_train_veh + n_val_veh]
    test_veh_idx = vehicle_indices[n_train_veh + n_val_veh:]

    # ── Step 2: 计算每辆车在 X 中的起始和结束行号 ──
    # np.cumsum 累加算出边界，前面补 0
    # 例如 track_lengths=[50, 120] -> bounds=[0, 50, 170]
    bounds = np.insert(np.cumsum(track_lengths), 0, 0)

    # ── Step 3: 根据划分好的车辆，收集它们对应的样本行号 ──
    train_idx, val_idx, test_idx = [], [], []

    for v_idx in train_veh_idx:
        train_idx.extend(range(bounds[v_idx], bounds[v_idx+1]))
        
    for v_idx in val_veh_idx:
        val_idx.extend(range(bounds[v_idx], bounds[v_idx+1]))
        
    for v_idx in test_veh_idx:
        test_idx.extend(range(bounds[v_idx], bounds[v_idx+1]))

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    total_samples = len(X)
    
    # ── Lazy Dataset ──
    train_dataset = TrajectoryDataset(X, y, train_idx)
    val_dataset = TrajectoryDataset(X, y, val_idx)
    test_dataset = TrajectoryDataset(X, y, test_idx)

    # ── Test 集加载到内存 ──
    X_test = np.array(X[test_idx])
    y_test = np.array(y[test_idx])
    
    X_train, y_train = X, y
    X_val, y_val = X, y

    # 打印信息 (因为车辆轨迹长度不一，样本比例会和车辆比例略有偏差，这是正常的)
    print(f"\n[create_dataloaders] 按【独立车辆】划分数据:")
    print(f"  总车辆数: {num_vehicles:,} 辆")
    print(f"  训练集: {len(train_veh_idx):,} 辆车 -> {len(train_idx):,} 样本 ({len(train_idx)/total_samples*100:.1f}%)")
    print(f"  验证集: {len(val_veh_idx):,} 辆车 -> {len(val_idx):,} 样本 ({len(val_idx)/total_samples*100:.1f}%)")
    print(f"  测试集: {len(test_veh_idx):,} 辆车 -> {len(test_idx):,} 样本 ({len(test_idx)/total_samples*100:.1f}%)\n")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader, test_loader,X_train, X_val, X_test,y_train, y_val, y_test
