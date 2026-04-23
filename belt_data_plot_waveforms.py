import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ==================== 配置 ====================
PROCESSED_DIR = r"E:\Processed"
FIGURE_ROOT = r"E:\Processed\Figures"

MODE_DISPLAY = {
    'eupnea': 'Eupnea (370s)',
    'apnea_5': 'Apnea 5s (205s)',
    'apnea_10': 'Apnea 10s (200s)',
    'apnea_15': 'Apnea 15s (195s)',
    'apnea_20': 'Apnea 20s (195s)'
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 辅助函数 ====================
def parse_filename(filename):
    base = filename.replace('_processed_30Hz.txt', '')
    pattern = r'(\d+)_\d+_([a-zA-Z]+(?:_\d+)?)_\d{8}_\d{6}'
    match = re.match(pattern, base)
    if match:
        return match.group(1), match.group(2), base
    return None, None, None

def load_signal(filepath):
    """加载信号，跳过第一行标题"""
    data = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 1]

def plot_single_waveform(t, sig, title, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(t, sig, 'b-', linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_participant_waveforms(participant_id, signals_info, save_dir):
    n = len(signals_info)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
    
    # 统一将 axes 转换为一维列表
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (mode, t, sig, basename) in enumerate(signals_info):
        ax = axes[idx]
        ax.plot(t, sig, linewidth=0.6)
        ax.set_title(f"{MODE_DISPLAY.get(mode, mode)}\n{basename}", fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # 隐藏多余的子图
    for idx in range(len(signals_info), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Participant {participant_id} - All Breathing Waveforms", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    save_path = os.path.join(save_dir, f"participant_{participant_id}_all_waveforms.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存参与者图: {save_path}")

def plot_mode_waveforms(mode, signals_info, save_dir):
    n = len(signals_info)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (participant_id, t, sig, basename) in enumerate(signals_info):
        ax = axes[idx]
        ax.plot(t, sig, linewidth=0.6)
        ax.set_title(f"Participant {participant_id}\n{basename}", fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    for idx in range(len(signals_info), len(axes)):
        axes[idx].axis('off')
    
    mode_display = MODE_DISPLAY.get(mode, mode)
    fig.suptitle(f"Mode: {mode_display} - All Participants", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    save_path = os.path.join(save_dir, f"mode_{mode}_all_waveforms.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存模式图: {save_path}")

# ==================== 主流程 ====================
def main():
    single_dir = os.path.join(FIGURE_ROOT, 'single')
    participant_dir = os.path.join(FIGURE_ROOT, 'participant')
    mode_dir = os.path.join(FIGURE_ROOT, 'mode')
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(participant_dir, exist_ok=True)
    os.makedirs(mode_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_processed_30Hz.txt')]
    print(f"找到 {len(all_files)} 个处理后的数据文件")
    
    participant_data = {}
    mode_data = {}
    
    for filename in all_files:
        filepath = os.path.join(PROCESSED_DIR, filename)
        participant, mode, basename = parse_filename(filename)
        if participant is None:
            print(f"警告：无法解析文件名 {filename}，跳过")
            continue
        
        t, sig = load_signal(filepath)
        
        # 单个波形图
        title = f"{basename}\nMode: {MODE_DISPLAY.get(mode, mode)} | Participant: {participant}"
        save_path = os.path.join(single_dir, f"{basename}_waveform.png")
        plot_single_waveform(t, sig, title, save_path)
        print(f"已保存单文件图: {save_path}")
        
        participant_data.setdefault(participant, []).append((mode, t, sig, basename))
        mode_data.setdefault(mode, []).append((participant, t, sig, basename))
    
    print("\n生成参与者分组图...")
    for participant_id, info_list in participant_data.items():
        plot_participant_waveforms(participant_id, info_list, participant_dir)
    
    print("\n生成实验模式分组图...")
    for mode, info_list in mode_data.items():
        plot_mode_waveforms(mode, info_list, mode_dir)
    
    print("\n所有图片生成完毕！")
    print(f"图片保存在: {FIGURE_ROOT}")

if __name__ == "__main__":
    main()