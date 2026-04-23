"""
split_dataset.py
按被试进行 80/10/10 划分，保存元数据 CSV 和对应的 video_labels、breath_signals。
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============== 路径配置 ==============
SAMPLES_DIR = "E:/Data/Samples"
META_CSV = f"{SAMPLES_DIR}/samples_metadata.csv"
DATA_NPZ = f"{SAMPLES_DIR}/samples_data.npz"

OUTPUT_DIR = SAMPLES_DIR  # 输出到同目录

RANDOM_SEED = 42

def main():
    # 1. 加载原始数据
    meta = pd.read_csv(META_CSV)
    data = np.load(DATA_NPZ)
    video_labels = data['video_labels']
    breath_signals = data['breath_signals']

    # 2. 按被试划分
    subjects = meta['subject_id'].unique()
    print(f"总被试数: {len(subjects)}")

    # 80% train, 20% temp (val + test)
    train_subjs, temp_subjs = train_test_split(subjects, test_size=0.2, random_state=RANDOM_SEED)
    # 将 temp 均分为 val 和 test (各10%)
    val_subjs, test_subjs = train_test_split(temp_subjs, test_size=0.5, random_state=RANDOM_SEED)

    # 3. 生成掩码
    train_mask = meta['subject_id'].isin(train_subjs)
    val_mask   = meta['subject_id'].isin(val_subjs)
    test_mask  = meta['subject_id'].isin(test_subjs)

    # 4. 提取各子集元数据（重置索引，保持连续）
    train_meta = meta[train_mask].reset_index(drop=True)
    val_meta   = meta[val_mask].reset_index(drop=True)
    test_meta  = meta[test_mask].reset_index(drop=True)

    # 5. 提取对应的 labels 和 breath
    train_labels = video_labels[train_mask]
    val_labels   = video_labels[val_mask]
    test_labels  = video_labels[test_mask]

    train_breath = breath_signals[train_mask]
    val_breath   = breath_signals[val_mask]
    test_breath  = breath_signals[test_mask]

    # 6. 保存
    train_meta.to_csv(f"{OUTPUT_DIR}/train_metadata.csv", index=False)
    val_meta.to_csv(f"{OUTPUT_DIR}/val_metadata.csv", index=False)
    test_meta.to_csv(f"{OUTPUT_DIR}/test_metadata.csv", index=False)

    np.savez_compressed(f"{OUTPUT_DIR}/train_data.npz",
                        video_labels=train_labels, breath_signals=train_breath)
    np.savez_compressed(f"{OUTPUT_DIR}/val_data.npz",
                        video_labels=val_labels, breath_signals=val_breath)
    np.savez_compressed(f"{OUTPUT_DIR}/test_data.npz",
                        video_labels=test_labels, breath_signals=test_breath)

    print("保存完成：")
    print(f"训练集: 样本数 {len(train_meta)}")
    print(f"验证集: 样本数 {len(val_meta)}")
    print(f"测试集: 样本数 {len(test_meta)}")

if __name__ == "__main__":
    main()