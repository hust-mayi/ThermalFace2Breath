"""
thermal_breath_dataset.py
自定义 Dataset，用于加载热成像视频片段、帧级标签和呼吸带信号。
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ThermalBreathDataset(Dataset):
    def __init__(self, meta_csv, data_npz, video_root, transform=None):
        """
        参数:
            meta_csv: 元数据 CSV 文件路径 (如 train_metadata.csv)
            data_npz: 对应的数据文件路径 (如 train_data.npz)，包含 video_labels 和 breath_signals
            video_root: 存放 .avi 视频的根目录
            transform: 可选，视频帧变换（如 torchvision.transforms）
        """
        self.meta = pd.read_csv(meta_csv)
        data = np.load(data_npz)
        self.video_labels = data['video_labels']      # shape: (N, 150)
        self.breath_signals = data['breath_signals']  # shape: (N, 900)
        self.video_root = video_root
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # 构建视频文件路径（根据你的命名规则调整）
        video_name = f"subject_{row['subject_id']}_session_{row['session_id']}_{row['mode']}.avi"
        video_path = os.path.join(self.video_root, video_name)

        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        num_frames = end_frame - start_frame + 1  # 应为 150

        # 读取视频片段
        cap = cv2.VideoCapture(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # 转为灰度图 (H, W)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()

        # 若读取帧数不足，补零
        while len(frames) < num_frames:
            frames.append(np.zeros_like(frames[0]) if frames else np.zeros((480, 640), dtype=np.uint8))
        video = np.stack(frames, axis=0)  # (150, H, W)

        if self.transform:
            video = self.transform(video)

        # 获取标签和呼吸信号
        labels = self.video_labels[idx].astype(np.int64)       # (150,)
        breath = self.breath_signals[idx].astype(np.float32)   # (900,)

        # 转为 tensor
        video_tensor = torch.tensor(video, dtype=torch.float32).unsqueeze(0)  # (1, 150, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        breath_tensor = torch.tensor(breath, dtype=torch.float32)

        return video_tensor, labels_tensor, breath_tensor