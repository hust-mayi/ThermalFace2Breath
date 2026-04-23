import os
import re
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = r"E:\RawVideo"
OUTPUT_DIR = r"E:\Processed"

DURATION_MAP = {
    'eupnea': 370,
    'apnea_5': 205,
    'apnea_10': 200,
    'apnea_15': 195,
    'apnea_20': 195,
}

HIGH_FS = 200.0
TARGET_FS = 30.0
LOWCUT = 0.1
HIGHCUT = 15.0
PAD_TIME = 2.0

# 去尖峰参数
MEDFILT_KERNEL = 5      # 中值滤波窗口大小（奇数）
CLIP_THRESHOLD_SIGMA = 4.0  # 限幅阈值（标准差倍数，设为0则不禁用）

def extract_mode_and_time(filename):
    pattern = r'(\d+_\d+)_([a-zA-Z]+(?:_\d+)?)_(\d{8}_\d{6})\.txt'
    match = re.match(pattern, filename)
    if match:
        prefix = match.group(1)
        mode = match.group(2)
        start_time = match.group(3)
        return mode, start_time, prefix
    return None, None, None

def pad_signal(signal, pad_len, mode='reflect'):
    if mode == 'reflect':
        pad_front = signal[1:pad_len+1][::-1]
        pad_back = signal[-pad_len-1:-1][::-1]
        return np.concatenate([pad_front, signal, pad_back])
    else:
        return np.pad(signal, pad_len, mode='edge')

def remove_spikes(signal, kernel_size=5, threshold_sigma=4.0):
    """
    去除信号中的尖锐峰
    1. 中值滤波平滑孤立尖峰
    2. 可选：将超出均值±threshold_sigma*std的值限幅
    """
    # 中值滤波
    signal_med = medfilt(signal, kernel_size=kernel_size)
    
    # 如果阈值有效，则进行限幅（避免将正常呼吸波削平）
    if threshold_sigma > 0:
        std = np.std(signal_med)
        threshold = threshold_sigma * std
        # 使用中值滤波后的信号作为参考来限幅原始信号
        # 但为保留细节，对原始信号限幅
        signal_clipped = np.clip(signal_med, -threshold, threshold)
        return signal_clipped
    else:
        return signal_med

def process_single_file(file_path, duration, high_fs, target_fs, lowcut, highcut, pad_time):
    data = np.loadtxt(file_path)
    n = len(data)
    if n == 0:
        raise ValueError("文件为空")
    original_fs = n / duration
    t_original = np.linspace(0, duration, n, endpoint=False)
    t_high = np.arange(0, duration, 1.0/high_fs)
    cs = CubicSpline(t_original, data, bc_type='natural')
    data_high = cs(t_high)
    pad_len = int(pad_time * high_fs)
    data_high_padded = pad_signal(data_high, pad_len, mode='reflect')
    nyquist = high_fs / 2.0
    b, a = butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
    data_high_padded_filtered = filtfilt(b, a, data_high_padded)
    data_high_filtered = data_high_padded_filtered[pad_len:-pad_len]
    dec_factor = high_fs / target_fs
    if abs(dec_factor - round(dec_factor)) < 1e-6:
        step = int(round(dec_factor))
        t_target = t_high[::step]
        signal_final = data_high_filtered[::step]
    else:
        t_target = np.arange(0, duration, 1.0/target_fs)
        cs2 = CubicSpline(t_high, data_high_filtered, bc_type='natural')
        signal_final = cs2(t_target)
    
    # ----- 去除尖峰 -----
    signal_final = remove_spikes(signal_final, 
                                 kernel_size=MEDFILT_KERNEL,
                                 threshold_sigma=CLIP_THRESHOLD_SIGMA)
    
    return t_target, signal_final

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"在 {INPUT_DIR} 中找到 {len(all_files)} 个 txt 文件")
    processed_count = 0
    skipped_count = 0
    for filename in all_files:
        mode, start_time, prefix = extract_mode_and_time(filename)
        if mode is None:
            print(f"跳过无法解析的文件: {filename}")
            skipped_count += 1
            continue
        if mode not in DURATION_MAP:
            print(f"未知实验模式 '{mode}'，跳过文件: {filename}")
            skipped_count += 1
            continue
        duration = DURATION_MAP[mode]
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            print(f"处理: {filename} (模式={mode}, 时长={duration}s)")
            t, sig = process_single_file(file_path, duration,
                                         HIGH_FS, TARGET_FS,
                                         LOWCUT, HIGHCUT, PAD_TIME)
            base_name = filename.replace('.txt', '')
            out_filename = f"{base_name}_processed_30Hz.txt"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            np.savetxt(out_path, np.column_stack((t, sig)),
                       header="Time(s) Amplitude", comments='')
            processed_count += 1
            print(f"  已保存: {out_filename}")
        except Exception as e:
            print(f"  处理失败: {filename}, 错误: {e}")
            skipped_count += 1
    print(f"\n批量处理完成: 成功处理 {processed_count} 个文件，跳过 {skipped_count} 个")

if __name__ == "__main__":
    main()