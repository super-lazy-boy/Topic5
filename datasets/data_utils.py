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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    批量加载多个录像的轨迹数据并拼接。
    """
    all_tracks = []
    all_recordingmeta = []
    for rid in rec_ids:
        tracks, recordingmeta_df = load_recording(data_path, rid)
        # 为了区分不同录像的车辆，给 "id" 列添加前缀
        tracks["id"] = str(rid) + "_" + tracks["id"].astype(str)
        all_tracks.append(tracks)
        all_recordingmeta.append(recordingmeta_df)
    return pd.DataFrame(pd.concat(all_tracks, ignore_index=True)), pd.DataFrame(pd.concat(all_recordingmeta, ignore_index=True))



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
    positions: np.ndarray,      # Shape: (seq_len, 2)  ← [x, y]
    velocities: np.ndarray,     # Shape: (seq_len, 2)  ← [vx, vy]
    accelerations: np.ndarray,  # Shape: (seq_len, 2)  ← [ax, ay]
    scalars: np.ndarray,        # Shape: (seq_len, 5)  ← [视距, dhw, thw, ttc 等]
    ref_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将全局坐标系下的多维特征转换到以目标车辆为中心的局部坐标系。
    """
    # ── Step 1: 平移 (仅针对位置) ──
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
    local_positions = translated_pos @ R.T

    # 对速度做旋转（速度是向量，不需要平移）
    local_velocities = velocities @ R.T
    
    # 对加速度做旋转（加速度同样是向量，不需要平移）
    local_accelerations = accelerations @ R.T

    # ── Step 4: 标量特征透传 ──
    # 距离、时间等标量在局部坐标系中保持不变
    local_scalars = scalars.copy()

    return (
        local_positions.astype(np.float32), 
        local_velocities.astype(np.float32),
        local_accelerations.astype(np.float32),
        local_scalars.astype(np.float32)
    )


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


def create_sequences(tracks_df: pd.DataFrame,
                    history_len: int = 20,
                    future_len: int = 30,
                    min_track_len: int = 60,
                    use_agent_centric: bool = True) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[int]]:
    
    window_len = history_len + future_len  # 总窗口 = 20 + 30 = 50 帧
    id_column = "id"

    # 统计每辆车有多少帧数据
    track_lengths = tracks_df.groupby(id_column).size()
    valid_ids = track_lengths[track_lengths >= min_track_len].index
    valid_df = tracks_df[tracks_df[id_column].isin(valid_ids)].copy()

    # 按车辆唯一标识分组，每组是该车按帧排序的完整轨迹
    grouped = valid_df.groupby(id_column)

    # 确定特征列和交互列
    base_features = ["x", "y", "xVelocity", "yVelocity","xAcceleration", "yAcceleration","frontSightDistance","backSightDistance","dhw","thw","ttc"]
    feature_dim = len(base_features)  

    history_list: List[np.ndarray] = []
    future_list: List[np.ndarray] = []
    track_lengths = []

    for track_id, track_group in grouped:
        # 按帧号排序
        track_group = track_group.sort_values("frame")
        # 提取特征列 
        global_values = track_group[base_features].values.astype(np.float32)
        # 查看看当前 track 的长度，计算可用窗口数
        track_len = global_values.shape[0]
          
        num_windows = track_len - window_len + 1
        track_lengths.append(num_windows) 

        # 收集当前 track 所有窗口的基础特征 
        track_history_base: List[np.ndarray] = []
        track_future_base: List[np.ndarray] = []

        if use_agent_centric: # 如果开启以自己车为中心
            for w_start in range(num_windows):
                # 按列索引截取对应特征
                w_pos = global_values[w_start : w_start + window_len, 0:2]  # 不copy
                w_vel = global_values[w_start : w_start + window_len, 2:4]
                w_acc = global_values[w_start : w_start + window_len, 4:6]
                w_sca = global_values[w_start : w_start + window_len, 6:11]

                # 转换到以当前车为中心的局部坐标系
                lp, lv, la, ls = transform_to_agent_centric(
                    w_pos, w_vel, w_acc, w_sca,
                    ref_idx=history_len - 1
                )
                lw = np.concatenate([lp, lv, la, ls], axis=-1)
                track_history_base.append(lw[:history_len])
                track_future_base.append(lw[history_len:])
        else:
            
            for start in range(num_windows):
                hist_end = start + history_len
                fut_end = hist_end + future_len
                track_history_base.append(global_values[start:hist_end])
                track_future_base.append(global_values[hist_end:fut_end])

        # 堆叠当前 track 的基础特征
        # (num_windows, history_len, 4) 和 (num_windows, future_len, 4)
        track_history = np.stack(track_history_base, axis=0).astype(np.float32)
        track_future = np.stack(track_future_base, axis=0).astype(np.float32)

        history_list.append(track_history)
        future_list.append(track_future)

    feature_dim = history_list[0].shape[-1]  
    # 计算总样本数
    total_samples = sum(arr.shape[0] for arr in history_list)

    history_all = np.concatenate(history_list, axis=0)
    future_all = np.concatenate(future_list, axis=0)

    history_list.clear()
    future_list.clear()

    history_flat = history_all.reshape(-1, feature_dim)
    future_flat = future_all.reshape(-1, feature_dim)

    combined_flat = np.concatenate([history_flat, future_flat], axis=0)

    scaler = StandardScaler()
    scaler.fit(combined_flat)  # 一键计算出整个数据集的 mean 和 std
    scaler.scale_ = np.maximum(scaler.scale_, 1e-10)

    history_flat_norm = scaler.transform(history_flat)
    future_flat_norm = scaler.transform(future_flat)

    history_norm = history_flat_norm.reshape(history_all.shape).astype(np.float32)
    future_norm = future_flat_norm.reshape(future_all.shape).astype(np.float32)

    print(f"[create_sequences] 生成了 {history_all.shape[0]:,} 个样本")
    print(f"  X shape: {history_norm.shape}")
    print(f"  y shape: {future_norm.shape}")

    return history_norm, future_norm, scaler, track_lengths
