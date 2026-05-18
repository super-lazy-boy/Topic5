import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


def load_recording(data_path: str, rec_id: int) :
    
    prefix = f"{rec_id:02d}"  
    
    tracks_df = pd.read_csv(Path(data_path) / f"{prefix}_tracks.csv")
    tracksMeta_df = pd.read_csv(Path(data_path)/ f"{prefix}_tracksMeta.csv")
    recordingmeta_df = pd.read_csv(Path(data_path) / f"{prefix}_recordingMeta.csv")

    # 找出同名列，合并时防止重复列冲突
    common_cols = set(tracks_df.columns).intersection(set(tracksMeta_df.columns))
    common_cols.discard("id")
    tracksMeta_df = tracksMeta_df.drop(columns=common_cols)
    tracks = pd.merge(tracks_df, tracksMeta_df, on="id", how="left") 

    return tracks,  recordingmeta_df


def load_multiple_recordings(
    data_path: str, rec_ids: List[int]
) -> pd.DataFrame:
    """
    批量加载多个录像的轨迹数据并拼接。
    """
    all_tracks = []
    for rid in rec_ids:
        tracks, recordingmeta_df = load_recording(data_path, rid)
        tracks["recordingId"] = rid  # 标记来源录像，防止 id 冲突
        all_tracks.append(tracks)
    return pd.concat(all_tracks, ignore_index=True)


# ============================================================================
#  第二部分: MTR 核心 — 以目标车辆为中心的坐标系变换
# ============================================================================

def compute_heading_angle(vx: float, vy: float) -> float:
    """
    根据速度分量计算车辆航向角。
    """
    return np.arctan2(vy, vx)


def build_rotation_matrix(theta: float) -> np.ndarray:
    """
    构建 2D 旋转变换矩阵。
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)


def transform_to_agent_centric(
    positions: np.ndarray,   # Shape: (seq_len, 2)  ← [x, y]
    velocities: np.ndarray,  # Shape: (seq_len, 2)  ← [vx, vy]
    ref_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将全局坐标系下的位置和速度转换到以目标车辆为中心的局部坐标系。
    """
    # ── Step 1: 平移 ──
    # 取出参考帧的位置作为"新原点"
    ref_pos = positions[ref_idx].copy()  # Shape: (2,)  ← [x_ref, y_ref]
    # 所有帧减去参考帧位置 → 此时参考帧变为 (0, 0)
    translated_pos = positions - ref_pos  # Shape: (seq_len, 2)

    # ── Step 2: 计算航向角 ──
    ref_vx = velocities[ref_idx, 0]  # 参考帧的 x 方向速度
    ref_vy = velocities[ref_idx, 1]  # 参考帧的 y 方向速度
    theta = compute_heading_angle(ref_vx, ref_vy)

    # ── Step 3: 旋转变换 ──
    R = build_rotation_matrix(theta)  # Shape: (2, 2)

    # 对平移后的位置做旋转
    # translated_pos: (seq_len, 2),  R: (2, 2)
    # → local_positions: (seq_len, 2)
    local_positions = translated_pos @ R.T

    # 对速度做同样的旋转（速度不依赖原点，只旋转方向）
    # velocities: (seq_len, 2),  R: (2, 2)
    # → local_velocities: (seq_len, 2)
    local_velocities = velocities @ R.T

    return local_positions.astype(np.float32), local_velocities.astype(np.float32)


# ============================================================================
#  第三部分: MTR 风格 — 周围交互车辆特征提取
# ============================================================================

def _build_neighbor_lookup(rec_data: pd.DataFrame) -> dict:
    """
    构建 (id, frame) → (x, y, vx, vy) 的 O(1) 查找字典。

    使用 numpy 向量化取值 + 纯 Python 循环构建 dict，
    比 iterrows() 快约 10-20 倍（避免每行构造 Series 对象）。
    对于 50 万行数据，构建耗时约 0.3-0.5 秒。
    """
    ids = rec_data["id"].values
    frames = rec_data["frame"].values
    xs = rec_data["x"].values
    ys = rec_data["y"].values
    vxs = rec_data["xVelocity"].values
    vys = rec_data["yVelocity"].values
    lookup = {}
    for i in range(len(ids)):
        lookup[(ids[i], frames[i])] = (float(xs[i]), float(ys[i]),
                                        float(vxs[i]), float(vys[i]))
    return lookup


def extract_interaction_features(
    target_track: pd.DataFrame,
    history_len: int,
    future_len: int,
    neighbor_lookup: dict,
) -> np.ndarray:
    """
    提取目标车辆周围的关键交互车辆特征（纯 numpy 向量化版）。

    === 性能优化 (v4 — 纯 numpy) ===
    1. neighbor_lookup 由外部一次性构建，所有 track 复用 — O(1) 邻居查找
    2. 所有 target 数据预提取为 numpy 数组 — 无 iterrows()/dict 开销
    3. 旋转计算内联化 — 避免每个 frame 构造 numpy 矩阵
    4. 预分配输出数组 — 一次性写入，无 list append

    === 特征维度 (24 维) ===
      每个方向 4 维 (relative_x, relative_y, relative_vx, relative_vy)
      6 方向 = preceding, following, leftPreceding, leftFollowing,
               rightPreceding, rightFollowing
    """
    total_len = history_len + future_len
    track_len = len(target_track)
    num_windows = track_len - total_len + 1

    if num_windows <= 0:
        return np.empty((0, total_len, 24), dtype=np.float32)

    # ── 一次性提取所有列为 numpy 数组（避免 iterrows / pandas 索引开销）──
    t_x = target_track["x"].values.astype(np.float32)
    t_y = target_track["y"].values.astype(np.float32)
    t_vx = target_track["xVelocity"].values.astype(np.float32)
    t_vy = target_track["yVelocity"].values.astype(np.float32)
    t_frames = target_track["frame"].values

    neighbor_dirs = [
        "precedingId", "followingId", "leftPrecedingId",
        "leftFollowingId", "rightPrecedingId", "rightFollowingId",
    ]
    # 预提取所有 neighbor ID 列（不存在的列填 0）
    neighbor_cols = []
    for col in neighbor_dirs:
        if col in target_track.columns:
            neighbor_cols.append(target_track[col].fillna(0).values.astype(np.int64))
        else:
            neighbor_cols.append(np.zeros(track_len, dtype=np.int64))

    # ── 预分配输出 ──
    result = np.zeros((num_windows, total_len, 24), dtype=np.float32)

    for w_idx in range(num_windows):
        for f_idx in range(total_len):
            pos = w_idx + f_idx  # track 级索引

            tx, ty = t_x[pos], t_y[pos]
            tvx, tvy = t_vx[pos], t_vy[pos]
            frame = int(t_frames[pos])

            # 内联旋转计算: 避免创建 (2,2) 矩阵和 matmul
            # R = [[cos, sin], [-sin, cos]],  rel @ R.T
            # rpx = dx*cos + dy*sin,  rpy = -dx*sin + dy*cos
            theta = np.arctan2(tvy, tvx)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            for d_idx in range(6):
                nid = int(neighbor_cols[d_idx][pos])
                if nid == 0:
                    continue

                n_data = neighbor_lookup.get((nid, frame))
                if n_data is None:
                    continue

                nx, ny, nvx, nvy = n_data
                dx = nx - tx
                dy = ny - ty

                off = d_idx * 4
                result[w_idx, f_idx, off] = dx * cos_t + dy * sin_t       # rel_x
                result[w_idx, f_idx, off + 1] = -dx * sin_t + dy * cos_t  # rel_y
                result[w_idx, f_idx, off + 2] = nvx * cos_t + nvy * sin_t # rel_vx
                result[w_idx, f_idx, off + 3] = -nvx * sin_t + nvy * cos_t# rel_vy

    return result


# ============================================================================
#  第四部分: 轨迹序列构建 (滑动窗口 + 坐标系变换 + 标准化)
# ============================================================================

def create_sequences(
    tracks_df: pd.DataFrame,
    history_len: int = 20,
    future_len: int = 30,
    min_track_len: int = 60,
    use_agent_centric: bool = True,
    use_interaction: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    window_len = history_len + future_len  # 总窗口 = 20 + 30 = 50 帧

    # ──── 流程 1: 过滤短轨迹 ────
    #
    # 【重要】若 DataFrame 中包含 "uniqueId" 列，则优先用该列做车辆标识；
    # 否则使用默认的 "id" 列。这是因为不同 highD 录像的车辆 id 会重置
    # (录像 01 的 id=1 和录像 02 的 id=1 是不同的车)，直接用 "id" 合并
    # 多个录像会导致车辆混淆。
    id_column = "uniqueId" if "uniqueId" in tracks_df.columns else "id"

    # 统计每辆车有多少帧数据
    track_lengths = tracks_df.groupby(id_column).size()
    # 只保留帧数足够的车辆
    valid_ids = track_lengths[track_lengths >= min_track_len].index
    valid_df = tracks_df[tracks_df[id_column].isin(valid_ids)].copy()
    # 此时 valid_df 丢弃了约 10~20% 的短轨迹车辆
    # Shape: valid_df 略小于 tracks_df

    # ──── 流程 2: 按车分组 ────
    # 按车辆唯一标识分组，每组是该车按帧排序的完整轨迹
    grouped = valid_df.groupby(id_column)

    # 确定特征列和交互列
    base_features = ["x", "y", "xVelocity", "yVelocity"]
    feature_dim = len(base_features)  # 4

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    # ── 预构建邻居查找字典（所有 track 共享，避免重复扫描）──
    if use_interaction:
        neighbor_lookups = {}
        if "recordingId" in tracks_df.columns:
            for rid, rec_data in tracks_df.groupby("recordingId"):
                neighbor_lookups[rid] = _build_neighbor_lookup(rec_data)
                print(f"  [交互索引] 录像 {rid}: {len(neighbor_lookups[rid]):,} 条记录已索引")
        else:
            neighbor_lookups[None] = _build_neighbor_lookup(tracks_df)

    for track_id, track_group in grouped:
        # 按帧号排序，确保时间顺序正确
        track_group = track_group.sort_values("frame")

        # 提取基础特征列 (全局坐标系)
        # Shape: (track_len, 4)  ← [x, y, xVelocity, yVelocity]
        global_values = track_group[base_features].values.astype(np.float32)
        track_len = global_values.shape[0]
        num_windows = track_len - window_len + 1

        # ── 收集当前 track 所有窗口的基础特征 ──
        track_X_base: List[np.ndarray] = []
        track_y_base: List[np.ndarray] = []

        # ──── 流程 3: 坐标系变换 (MTR 核心) ────
        if use_agent_centric:
            # 逐窗口进行坐标变换。
            # 为什么逐窗口而不是整条轨迹一次变换？
            # → 因为每个窗口的"参考帧"(历史段最后一帧)不同：
            #   窗口 [0:50] 以第 19 帧为参考，窗口 [1:51] 以第 20 帧为参考。
            #   不同参考帧对应不同的目标车位置和航向角，
            #   因此每个窗口需要独立做平移+旋转变换。
            for w_start in range(num_windows):
                # 截取当前窗口的全局坐标
                # Shape: (window_len, 2)  ← [x, y]
                w_pos = global_values[w_start : w_start + window_len, :2].copy()
                # Shape: (window_len, 2)  ← [vx, vy]
                w_vel = global_values[w_start : w_start + window_len, 2:4].copy()

                # MTR 核心变换: 以历史段最后一帧为参考帧
                #   - 平移到参考帧位置 → 参考帧变为 (0, 0)
                #   - 旋转使参考帧航向对齐 +X 轴
                # Shape: 都是 (window_len, 2)
                lp, lv = transform_to_agent_centric(
                    w_pos, w_vel,
                    ref_idx=history_len - 1  # 固定取窗口内第 history_len-1 帧(=历史末帧)
                )

                # 拼接位置和速度 → 4 维特征向量
                lw = np.concatenate([lp, lv], axis=-1)  # (window_len, 4)

                # 切分为历史段(输入)和未来段(标签)
                track_X_base.append(lw[:history_len].copy())   # (history_len, 4)
                track_y_base.append(lw[history_len:].copy())   # (future_len, 4)
        else:
            # ──── 流程 4: 滑动窗口 (不使用 agent_centric 时的回退方案) ────
            for start in range(num_windows):
                hist_end = start + history_len
                fut_end = hist_end + future_len
                track_X_base.append(global_values[start:hist_end].copy())
                track_y_base.append(global_values[hist_end:fut_end].copy())

        # 堆叠当前 track 的基础特征
        # (num_windows, history_len, 4) 和 (num_windows, future_len, 4)
        track_X = np.stack(track_X_base, axis=0).astype(np.float32)
        track_y = np.stack(track_y_base, axis=0).astype(np.float32)

        # ──── 流程 4.5: 交互特征 (MTR 风格 — 新增) ────
        if use_interaction:
            # extract_interaction_features 对整条 track 一次性提取所有窗口的交互特征
            # 内部使用与基础特征相同的滑动窗口逻辑（窗口数 = num_windows）
            # 6 个邻居方向 × 4 特征(相对x, 相对y, 相对vx, 相对vy) = 24 维
            # 取当前 track 所属录像的预构建查找字典
            rec_id = (track_group["recordingId"].iloc[0]
                      if "recordingId" in track_group.columns else None)
            lookup = neighbor_lookups.get(rec_id, neighbor_lookups.get(None))
            inter_features = extract_interaction_features(
                track_group,
                history_len=history_len,
                future_len=future_len,
                neighbor_lookup=lookup,  # 预构建的 O(1) 字典，所有 track 复用
            )
            # inter_features: (num_windows, history_len+future_len, 24)

            # 切分为历史段和未来段
            track_X_inter = inter_features[:, :history_len, :]   # (num_windows, history_len, 24)
            track_y_inter = inter_features[:, history_len:, :]    # (num_windows, future_len, 24)

            # 拼接基础特征 + 交互特征 → 28 维
            track_X = np.concatenate([track_X, track_X_inter], axis=-1)  # (num_windows, history_len, 28)
            track_y = np.concatenate([track_y, track_y_inter], axis=-1)  # (num_windows, future_len, 28)

        X_list.append(track_X)
        y_list.append(track_y)

    # ═════════════════════════════════════════════════════════════
    # 流程 5: 写入 memmap（避免 np.concatenate 的峰值内存）
    # ═════════════════════════════════════════════════════════════
    # 旧版: np.concatenate(X_list) 会在内存中分配 (total_samples, 20, 28)
    #       的巨型数组 (~7.6 GiB float32 / ~15 GiB float64)，导致 OOM。
    # 新版: 提前计算总样本数，创建磁盘 memmap，逐 track 写入，
    #       内存中只保留当前 track 的数据。

    feature_dim = X_list[0].shape[-1]  # 4 (仅基础) 或 28 (含交互)

    # 删除最后一个 track_X/track_y 引用（已被 X_list[-1] 持有）
    del track_X, track_y

    # 计算总样本数
    total_samples = sum(arr.shape[0] for arr in X_list)

    # 创建 memmap（磁盘文件，非内存）
    import os as _os
    data_dir = _os.path.dirname(_os.path.dirname(__file__))
    processed_dir = _os.path.join(data_dir, "data", "processed")
    _os.makedirs(processed_dir, exist_ok=True)
    X_path = _os.path.join(processed_dir, "X_norm.dat")
    y_path = _os.path.join(processed_dir, "y_norm.dat")

    X_mm = np.memmap(X_path, dtype=np.float32, mode="w+",
                     shape=(total_samples, history_len, feature_dim))
    y_mm = np.memmap(y_path, dtype=np.float32, mode="w+",
                     shape=(total_samples, future_len, feature_dim))

    # 逐 track 写入 memmap，写完后立即释放
    offset = 0
    for arr in X_list:
        n = arr.shape[0]
        X_mm[offset:offset + n] = arr
        offset += n
    X_list.clear()  # 释放所有 track 数组

    offset = 0
    for arr in y_list:
        n = arr.shape[0]
        y_mm[offset:offset + n] = arr
        offset += n
    y_list.clear()

    X_mm.flush()
    y_mm.flush()

    # ═════════════════════════════════════════════════════════════
    # 流程 6: 增量 StandardScaler（避免 np.concatenate(X, y)）
    # ═════════════════════════════════════════════════════════════
    # 旧版: np.concatenate([X_flat, y_flat]) 再分配 (total*50, 28)
    #       的大型数组（~5 GiB）。
    # 新版: 分别计算 X 和 y 的均值/方差，用增量公式合并。
    num_samples = total_samples
    n_x = num_samples * history_len
    n_y = num_samples * future_len
    n_total = n_x + n_y

    X_flat = X_mm.reshape(-1, feature_dim)  # memmap view, 0 内存
    y_flat = y_mm.reshape(-1, feature_dim)

    # 分别求均值
    mean_x = X_flat.mean(axis=0)
    mean_y = y_flat.mean(axis=0)
    # 增量合并均值
    combined_mean = (mean_x * n_x + mean_y * n_y) / n_total

    # 分别求方差 (population variance, ddof=0, 与 StandardScaler 一致)
    var_x = X_flat.var(axis=0) * n_x  # = sum of squared deviations
    var_y = y_flat.var(axis=0) * n_y
    # 增量合并方差 (Welford 并行算法)
    combined_var = (
        var_x + var_y
        + n_x * (mean_x - combined_mean) ** 2
        + n_y * (mean_y - combined_mean) ** 2
    ) / n_total

    # 构造 scaler（手动设置属性，与原 StandardScaler 行为一致）
    scaler = StandardScaler()
    scaler.mean_ = combined_mean
    scaler.var_ = combined_var
    scaler.scale_ = np.sqrt(combined_var)
    # 防止全零特征（如始终无邻居的交互方向）导致除零 → inf
    scaler.scale_ = np.maximum(scaler.scale_, 1e-10)

    # ── 就地标准化（直接写入 memmap，不创建新数组）──
    X_flat[:] = (X_flat - scaler.mean_) / scaler.scale_
    y_flat[:] = (y_flat - scaler.mean_) / scaler.scale_
    X_mm.flush()
    y_mm.flush()

    print(f"[create_sequences] 生成 {num_samples:,} 个样本")
    print(f"  X shape: {X_mm.shape}  (样本数, 历史帧数, 特征维数)")
    print(f"  y shape: {y_mm.shape}  (样本数, 未来帧数, 特征维数)")
    print(f"  存储路径: {processed_dir}")
    print(f"  Scaler mean (前2维): {scaler.mean_[:2]}")
    print(f"  Scaler var  (前2维): {scaler.var_[:2]}")

    return X_mm, y_mm, scaler
