"""
train.py
展示如何使用 ThermalBreathDataset 和 DataLoader 进行训练。
"""

import torch
from torch.utils.data import DataLoader
from thermal_breath_dataset import ThermalBreathDataset
import torch.nn as nn
import torch.optim as optim

# 路径配置
VIDEO_ROOT = "E:/Data/Video"
TRAIN_META = "E:/Data/Samples/train_metadata.csv"
TRAIN_DATA = "E:/Data/Samples/train_data.npz"
VAL_META   = "E:/Data/Samples/val_metadata.csv"
VAL_DATA   = "E:/Data/Samples/val_data.npz"

# 创建数据集
train_dataset = ThermalBreathDataset(TRAIN_META, TRAIN_DATA, VIDEO_ROOT)
val_dataset   = ThermalBreathDataset(VAL_META, VAL_DATA, VIDEO_ROOT)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# 示例模型（需根据你的任务定义）
class SimpleVideoRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: (batch, 1, 150, H, W)
        self.conv3d = nn.Conv3d(1, 16, kernel_size=(3, 7, 7), padding=(1, 3, 3))
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.fc = nn.Linear(16 * 75 * 60 * 80, 900)  # 假设 H=240, W=320，需计算实际尺寸

    def forward(self, x):
        x = self.pool(torch.relu(self.conv3d(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleVideoRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(10):
    model.train()
    for video, labels, breath in train_loader:
        video, breath = video.to(device), breath.to(device)
        optimizer.zero_grad()
        output = model(video)
        loss = criterion(output, breath)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for video, labels, breath in val_loader:
            video, breath = video.to(device), breath.to(device)
            output = model(video)
            val_loss += criterion(output, breath).item()
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")