import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============== 配置路径 ==============
ALIGNED_DIR = r"E:\Data\Aligned"          # 存放 aligned .npz 文件
BREATH_DIR = r"E:\Data\BreathBelt"        # 原始呼吸带 .txt 文件
OUTPUT_DIR = r"E:\Data\Samples"           # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 滑窗参数
WINDOW_SEC = 30
WINDOW_FRAMES = WINDOW_SEC * 5            # 150 帧
STEP_SEC = 5
STEP_FRAMES = STEP_SEC * 5                # 25 帧
BREATH_SAMPLES = WINDOW_SEC * 30          # 900 个采样点

# ============== 工具函数 ==============
def find_continuous_segments(condition):
    """找出 condition 为 True 的连续区间，返回 [(start, end), ...]"""
    segments = []
    in_seg = False
    start = 0
    for i, val in enumerate(condition):
        if val and not in_seg:
            start = i
            in_seg = True
        elif not val and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(condition) - 1))
    return segments

def extract_breath_signal(breath_file, start_idx, length):
    """从原始呼吸带文件中提取从 start_idx 开始的 length 个采样点"""
    data = np.loadtxt(breath_file, skiprows=1)
    end_idx = min(start_idx + length, len(data))
    signal = data[start_idx:end_idx, 1]
    if len(signal) < length:
        signal = np.pad(signal, (0, length - len(signal)), constant_values=np.nan)
    return signal.astype(np.float32)

# ============== 单个实验处理 ==============
def process_session(npz_path, breath_path, mode):
    """处理单个实验，返回该实验生成的所有窗口的元数据列表和数据数组"""
    data = np.load(npz_path)
    labels = data['label']
    breath_idx = data['breath_idx']
    total_frames = len(labels)

    windows_meta = []
    video_labels_list = []
    breath_signals_list = []

    if mode == 'eupnea':
        # 只取标签为1的连续段
        mask = (labels == 1)
        segments = find_continuous_segments(mask)
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len < WINDOW_FRAMES:
                continue
            # 在段内以步长 STEP_FRAMES 滑动
            for start in range(seg_start, seg_end - WINDOW_FRAMES + 2, STEP_FRAMES):
                end = start + WINDOW_FRAMES - 1
                # 二次确认窗口内全为1（理论上已在段内）
                if not np.all(labels[start:end+1] == 1):
                    continue
                # 提取呼吸带数据
                b_start = breath_idx[start]
                if b_start == -1:
                    continue
                b_signal = extract_breath_signal(breath_path, b_start, BREATH_SAMPLES)

                windows_meta.append({
                    'start_frame': start,
                    'end_frame': end,
                    'apnea_start': -1,
                    'apnea_end': -1
                })
                video_labels_list.append(labels[start:end+1])
                breath_signals_list.append(b_signal)

    else:  # apnea 系列
        # 只取标签为1或2的连续段
        mask = ((labels == 1) | (labels == 2))
        segments = find_continuous_segments(mask)
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start + 1
            if seg_len < WINDOW_FRAMES:
                continue
            # 在当前段内找出所有标签为2的连续子段
            sub_mask = (labels[seg_start:seg_end+1] == 2)
            sub_segments = find_continuous_segments(sub_mask)
            for sub_rel_start, sub_rel_end in sub_segments:
                abs_ap_start = seg_start + sub_rel_start
                abs_ap_end = seg_start + sub_rel_end
                ap_len = abs_ap_end - abs_ap_start + 1
                # 计算窗口起始范围：窗口必须完全包含该apnea段，且窗口在段边界内
                win_start_min = max(seg_start, abs_ap_end - WINDOW_FRAMES + 1)
                win_start_max = min(abs_ap_start, seg_end - WINDOW_FRAMES + 1)
                if win_start_min > win_start_max:
                    continue
                # 以步长生成所有符合条件的起始帧
                # 为保证窗口内标签只有1和2，已在段内保证
                for start in range(win_start_min, win_start_max + 1, STEP_FRAMES):
                    end = start + WINDOW_FRAMES - 1
                    b_start = breath_idx[start]
                    if b_start == -1:
                        continue
                    b_signal = extract_breath_signal(breath_path, b_start, BREATH_SAMPLES)

                    windows_meta.append({
                        'start_frame': start,
                        'end_frame': end,
                        'apnea_start': abs_ap_start,
                        'apnea_end': abs_ap_end
                    })
                    video_labels_list.append(labels[start:end+1])
                    breath_signals_list.append(b_signal)

    return windows_meta, video_labels_list, breath_signals_list

# ============== 主函数 ==============
def main():
    # 获取所有 aligned .npz 文件
    npz_files = [f for f in os.listdir(ALIGNED_DIR) if f.endswith('_aligned.npz')]
    if not npz_files:
        print("未找到 aligned .npz 文件。")
        return

    # 用于存储所有样本的列表
    all_meta = []
    all_video_labels = []
    all_breath_signals = []

    pattern = re.compile(r"(\d+)_(\d+)_(\w+)_aligned\.npz")
    for npz_file in tqdm(npz_files, desc="生成样本"):
        m = pattern.match(npz_file)
        if not m:
            continue
        subj, sess, mode = m.groups()
        npz_path = os.path.join(ALIGNED_DIR, npz_file)

        # 找到对应的原始呼吸带文件
        breath_name = f"{subj}_{sess}_{mode}_"
        breath_files = [f for f in os.listdir(BREATH_DIR) if f.startswith(breath_name) and f.endswith('_processed_30Hz.txt')]
        if not breath_files:
            print(f"警告: 找不到呼吸带文件对应 {subj}_{sess}_{mode}")
            continue
        breath_path = os.path.join(BREATH_DIR, breath_files[0])

        try:
            meta_list, vid_list, breath_list = process_session(npz_path, breath_path, mode)
        except Exception as e:
            print(f"处理 {npz_file} 时出错: {e}")
            continue

        # 附加 subject, session, mode 信息
        for m in meta_list:
            m['subject_id'] = subj
            m['session_id'] = sess
            m['mode'] = mode
            all_meta.append(m)
        all_video_labels.extend(vid_list)
        all_breath_signals.extend(breath_list)

    if not all_meta:
        print("未生成任何样本。")
        return

    # 保存元数据 CSV
    meta_df = pd.DataFrame(all_meta)
    meta_csv = os.path.join(OUTPUT_DIR, "samples_metadata.csv")
    meta_df.to_csv(meta_csv, index=False)
    print(f"样本元数据已保存至: {meta_csv}")

    # 保存数据 NPZ
    video_labels_arr = np.stack(all_video_labels, axis=0).astype(np.uint8)
    breath_signals_arr = np.stack(all_breath_signals, axis=0).astype(np.float32)
    data_npz = os.path.join(OUTPUT_DIR, "samples_data.npz")
    np.savez_compressed(data_npz,
                        video_labels=video_labels_arr,
                        breath_signals=breath_signals_arr)
    print(f"样本数据已保存至: {data_npz}")
    print(f"总计生成样本数: {len(all_meta)}")

if __name__ == "__main__":
    main()