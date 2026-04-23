import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ============== 配置路径 ==============
DATA_ROOT = r"E:\Data"
VIDEO_DIR = os.path.join(DATA_ROOT, "Video")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")
RECORD_DIR = os.path.join(DATA_ROOT, "Record")
BREATH_DIR = os.path.join(DATA_ROOT, "BreathBelt")

OUTPUT_DIR = os.path.join(DATA_ROOT, "Aligned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== 文件名正则匹配 ==============
VIDEO_PATTERN = re.compile(
    r"subject_(\d+)_session_(\d+)_(\w+)_(\d{8}-\d{9})\.avi$", re.I
)
LABEL_PATTERN = re.compile(
    r"subject_(\d+)_session_(\d+)_(\w+)_(\d{8}-\d{9})_labels\.npy$", re.I
)
RECORD_PATTERN = re.compile(
    r"subject_(\d+)_session_(\d+)_(\w+)_(\d{8}_\d{6})_completed_summary\.csv$", re.I
)
BREATH_PATTERN = re.compile(
    r"(\d+)_(\d+)_(\w+)_(\d{8}_\d{6})_processed_30Hz\.txt$", re.I
)

# ============== 时间解析函数 ==============
def parse_video_time(time_str: str) -> datetime:
    return datetime.strptime(time_str, "%Y%m%d-%H%M%S%f")

def parse_breath_time(time_str: str) -> datetime:
    return datetime.strptime(time_str, "%Y%m%d_%H%M%S")

# ============== 文件匹配 ==============
def match_files() -> dict:
    sessions = defaultdict(dict)

    for fname in os.listdir(VIDEO_DIR):
        m = VIDEO_PATTERN.match(fname)
        if not m:
            continue
        subj, sess, mode, time_str = m.groups()
        key = (subj, sess, mode)
        sessions[key]['video'] = os.path.join(VIDEO_DIR, fname)
        sessions[key]['video_time'] = parse_video_time(time_str)

    for fname in os.listdir(LABEL_DIR):
        m = LABEL_PATTERN.match(fname)
        if not m:
            continue
        subj, sess, mode, time_str = m.groups()
        key = (subj, sess, mode)
        sessions[key]['label'] = os.path.join(LABEL_DIR, fname)
        sessions[key]['label_time'] = parse_video_time(time_str)

    for fname in os.listdir(RECORD_DIR):
        m = RECORD_PATTERN.match(fname)
        if not m:
            continue
        subj, sess, mode, time_str = m.groups()
        key = (subj, sess, mode)
        sessions[key]['record'] = os.path.join(RECORD_DIR, fname)
        sessions[key]['record_end_time'] = datetime.strptime(time_str, "%Y%m%d_%H%M%S")

    for fname in os.listdir(BREATH_DIR):
        m = BREATH_PATTERN.match(fname)
        if not m:
            continue
        subj, sess, mode, time_str = m.groups()
        key = (subj, sess, mode)
        sessions[key]['breath'] = os.path.join(BREATH_DIR, fname)
        sessions[key]['breath_time'] = parse_breath_time(time_str)

    complete = {}
    for key, files in sessions.items():
        if all(k in files for k in ['video', 'label', 'record', 'breath']):
            complete[key] = files
        else:
            print(f"警告: 实验 {key} 文件缺失，已跳过。")
    return complete

# ============== 单次实验对齐 ==============
def align_experiment(video_path, label_path, record_path, breath_path,
                     video_start_dt, breath_start_dt):
    # 1. 读取CSV获取实验起止时间
    df = pd.read_csv(record_path)
    df.columns = df.columns.str.strip()

    prep_rows = df[df['phase_name'].str.lower().isin(['prep', 'preparation'])]
    if prep_rows.empty:
        prep_rows = df[df['category'].str.lower() == 'preparation']
    if prep_rows.empty:
        raise ValueError(f"CSV中未找到 prep 阶段，文件: {record_path}")

    end_rows = df[df['phase_name'].str.lower() == 'end']
    if end_rows.empty:
        end_rows = df[df['category'].str.lower() == 'end']
    if end_rows.empty:
        end_rows = df[df['display_name'].str.contains('结束', na=False)]
    if end_rows.empty:
        raise ValueError(f"CSV中未找到 end 阶段，文件: {record_path}")

    prep = prep_rows.iloc[0]
    end_phase = end_rows.iloc[0]

    exp_start = datetime.strptime(prep['actual_start_time'], "%Y-%m-%d %H:%M:%S.%f")
    exp_end = datetime.strptime(end_phase['actual_end_time'], "%Y-%m-%d %H:%M:%S.%f")

    # 2. 计算视频帧索引范围
    offset_video_start = (exp_start - video_start_dt).total_seconds()
    offset_video_end   = (exp_end - video_start_dt).total_seconds()
    frame_start = int(round(offset_video_start * 5))
    frame_end   = int(round(offset_video_end * 5))

    # 3. 计算呼吸带采样点索引范围
    offset_breath_start = (exp_start - breath_start_dt).total_seconds()
    offset_breath_end   = (exp_end - breath_start_dt).total_seconds()

    # 4. 读取标签数据
    labels_all = np.load(label_path)
    if labels_all.ndim == 2:
        labels_all = labels_all[:, 1]
    if frame_start < 0:
        frame_start = 0
    if frame_end >= len(labels_all):
        frame_end = len(labels_all) - 1
    exp_labels = labels_all[frame_start:frame_end+1]
    num_frames = len(exp_labels)

    # 5. 读取呼吸带数据（跳过第一行列名）
    belt_data = np.loadtxt(breath_path, skiprows=1)
    times = belt_data[:, 0]
    amplitudes = belt_data[:, 1]

    # 截取实验区间内的采样点
    mask = (times >= offset_breath_start) & (times <= offset_breath_end)
    valid_times = times[mask] - offset_breath_start  # 相对实验开始的时间
    valid_amps = amplitudes[mask]
    original_indices = np.where(mask)[0]  # 原始文件中的行索引（0-based，跳过表头后）

    # 6. 为每一帧计算聚合幅度和起始索引
    frame_amplitudes = np.full(num_frames, np.nan, dtype=np.float32)
    breath_start_indices = np.full(num_frames, -1, dtype=np.int32)

    for i in range(num_frames):
        t_start = i / 5.0
        t_end = (i + 1) / 5.0
        start_pos = np.searchsorted(valid_times, t_start, side='left')
        end_pos = np.searchsorted(valid_times, t_end, side='left')
        if start_pos < len(valid_times):
            breath_start_indices[i] = original_indices[start_pos]
        if start_pos < end_pos:
            frame_amplitudes[i] = np.mean(valid_amps[start_pos:end_pos])

    return {
        'frame_idx': np.arange(num_frames, dtype=np.uint32),
        'label': exp_labels.astype(np.uint8),
        'amplitude': frame_amplitudes,
        'breath_idx': breath_start_indices
    }

# ============== 批量处理与保存 ==============
def process_all_sessions():
    sessions = match_files()
    print(f"共找到 {len(sessions)} 个完整实验。")

    summary_rows = []
    for (subj, sess, mode), files in tqdm(sessions.items(), desc="处理实验"):
        try:
            data = align_experiment(
                video_path=files['video'],
                label_path=files['label'],
                record_path=files['record'],
                breath_path=files['breath'],
                video_start_dt=files['video_time'],
                breath_start_dt=files['breath_time']
            )
        except Exception as e:
            print(f"\n处理实验 {subj}_{sess}_{mode} 时出错: {e}")
            continue

        out_name = f"{subj}_{sess}_{mode}_aligned.npz"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        np.savez_compressed(out_path, **data)

        labels = data['label']
        total_frames = len(labels)
        label_counts = np.bincount(labels, minlength=3)
        summary_rows.append({
            'subject_id': subj,
            'session_id': sess,
            'mode': mode,
            'total_frames': total_frames,
            'label_0_count': label_counts[0],
            'label_1_count': label_counts[1],
            'label_2_count': label_counts[2],
            'npz_file': out_name
        })

    # 保存汇总CSV（带权限错误容错）
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUTPUT_DIR, "sessions_summary.csv")
        try:
            summary_df.to_csv(summary_csv, index=False)
            print(f"\n汇总表已保存至: {summary_csv}")
        except PermissionError:
            alt_csv = os.path.join(os.getcwd(), "sessions_summary.csv")
            summary_df.to_csv(alt_csv, index=False)
            print(f"\n警告: 无法写入 {summary_csv}，可能是文件被占用。")
            print(f"汇总表已保存至备用位置: {alt_csv}")
        except Exception as e:
            print(f"\n保存汇总表时出错: {e}")
    else:
        print("\n没有成功处理任何实验。")
    print(f"对齐数据已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_sessions()