import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============== 配置路径 ==============
INPUT_DIR = r"E:\Data\Aligned"          # 存放 .npz 文件的目录
OUTPUT_DIR = r"E:\Data\Aligned\CSV"     # 存放输出 .csv 的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_npz_to_csv(npz_path, csv_path):
    """
    将单个 .npz 文件转换为 CSV。
    期望 .npz 包含: frame_idx, label, amplitude, breath_idx (可选)
    """
    data = np.load(npz_path)

    # 必需键
    required_keys = ['frame_idx', 'label', 'amplitude']
    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"警告: {os.path.basename(npz_path)} 缺少必需键 {missing}，已跳过。")
        data.close()
        return

    # 构建 DataFrame
    df = pd.DataFrame({
        'frame_idx': data['frame_idx'],
        'label': data['label'],
        'amplitude': data['amplitude']
    })

    # 可选键 breath_idx：如果存在则添加，否则填充 -1
    if 'breath_idx' in data:
        df['breath_idx'] = data['breath_idx']
    else:
        print(f"提示: {os.path.basename(npz_path)} 不包含 breath_idx 列，将填充 -1。")
        df['breath_idx'] = -1

    # 保存为 CSV（不保存行索引）
    df.to_csv(csv_path, index=False)
    data.close()

def batch_convert():
    """批量转换所有 .npz 文件"""
    npz_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.npz')]
    if not npz_files:
        print(f"在 {INPUT_DIR} 中未找到 .npz 文件。")
        return

    print(f"找到 {len(npz_files)} 个 .npz 文件，开始转换...")
    for npz_file in tqdm(npz_files, desc="转换中"):
        npz_path = os.path.join(INPUT_DIR, npz_file)
        csv_name = npz_file.replace('.npz', '.csv')
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        try:
            convert_npz_to_csv(npz_path, csv_path)
        except Exception as e:
            print(f"\n转换 {npz_file} 时出错: {e}")

    print(f"\n所有转换完成，CSV 文件已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    batch_convert()