ThermalFace2Breath

1. 项目简介

项目目标：利用热红外面部视频估计呼吸信号，并进行呼吸暂停（apnea）检测。系统结合热红外视频与呼吸带基准信号，通过时间戳对齐构建监督学习数据集，最终从视频帧序列预测呼吸波形或分类呼吸状态。

多模态数据：热红外视频（.avi） + 呼吸带信号（.txt）

一句话概括 Pipeline：原始热像视频与呼吸带 → 标签生成 → 时间对齐 → 滑窗采样 → 数据集划分 → 模型训练。


2. 项目结构

process_thermal_video.py          热红外视频预处理（灰度反转、伽马校正）
batch_label_with_log.py           批量生成帧级标签（.npy）
belt_data_process.py              呼吸带原始信号处理（重采样、滤波、去尖峰）
align_data.py                     多模态数据时间对齐，输出对齐 .npz
visualize_labels.py               可视化视频与标签叠加
belt_data_plot_waveforms.py       绘制呼吸带波形图
generate_training_samples.py      构建训练样本（滑窗、输出 metadata+data）
split_dataset.py                  按被试划分训练/验证/测试集
thermal_breath_dataset.py         PyTorch Dataset 类
train.py                          训练示例
npy_to_csv.py                     工具：标签 .npy → .csv
npz_to_csv.py                     工具：对齐 .npz → .csv


3. 数据组织规范

3.1 视频命名规则
subject_{ID}_session_{SID}_{mode}_{YYYYMMDD}-{HHMMSSfff}.avi
例：subject_403_session_107_apnea_20_20260413-203148838.avi
mode：eupnea、apnea_5/10/15/20
时间戳：视频录制的起始时刻（精确到毫秒）

3.2 CSV 实验记录
每个视频对应一个 CSV，命名 subject_{ID}_session_{SID}_{mode}_{YYYYMMDD}_{HHMMSS}_completed_summary.csv
包含列：
  - experiment_start_time / experiment_end_time：实验总起止
  - 阶段行：phase_name, category, actual_start_time, actual_end_time
  - category 中 apnea_hold 代表屏气段，其他（如 preparation、eupnea）为正常呼吸段

3.3 标签文件（.npy）
{video_stem}_labels.npy：一维 int8 数组，长度 = 视频总帧数
  - 0：实验外（Before/After）
  - 1：正常呼吸（Normal Breathing）
  - 2：屏气（Apnea Hold）

3.4 呼吸带数据
   - 原始：{subject}_{session}_{mode}_{YYYYMMDD_HHMMSS}.txt
   - 处理后：{原始文件名}_processed_30Hz.txt
     第一行为注释 "Time(s) Amplitude"，之后每行：<时间(秒)> <幅度>

3.5 核心对齐原则
通过文件名中的时间戳，将视频起始时刻与呼吸带起始时刻统一到 UTC 时间，再利用 CSV 中的实验起止时间，截取对应帧和呼吸带采样点。


4. 数据处理流程

步骤概述（按顺序）：
1. 视频预处理（可选）：灰度 + 反转 + 伽马校正

2. 帧级标签生成：解析 CSV，为每一帧标注 0/1/2

3. 呼吸带信号处理：重采样至 200 Hz，带通滤波，降采样至 30 Hz，去尖峰

4. 多模态时间对齐：根据实验起止时间截取视频帧和呼吸带信号，生成 aligned .npz

5. 滑窗样本构建：30 秒窗口，步长 5 秒，提取视频标签序列和呼吸带片段

6. 数据集划分：按被试 80/10/10 划分

7. 训练：PyTorch Dataset 加载，模型自定义

   


5. 各模块说明

5.1 process_thermal_video.py
输入：E:/RawVideo/*.avi（原始彩色热像视频）
处理：灰度化 → 反转（负片） → 伽马校正（gamma=0.5）
输出：E:/ProcessedVideo/*.avi（单通道 8 位灰度视频）

5.2 batch_label_with_log.py
输入：视频目录 E:\RawVideo（含 .avi 和对应 CSV）
逻辑：
  1. 解析视频文件名获取起始时间与模式
  2. 匹配同名 CSV，提取实验起止时间和 apnea_hold 阶段
  3. 遍历每一帧，根据时间戳所属阶段赋予标签 0/1/2
  4. 优先使用视频实际帧率（cv2.CAP_PROP_FPS），否则默认 5 Hz
输出：E:\RawVideo\labels\{video_stem}_labels.npy，并记录日志

5.3 belt_data_process.py
输入：E:\RawVideo\{pattern}.txt（原始呼吸带信号）
处理：
  1. 根据 mode 查找固定时长（DURATION_MAP）
  2. 计算原始采样率 = len(data)/duration
  3. 立方插值重采样至 200 Hz，带通滤波（0.1–15 Hz）
  4. 降采样至 30 Hz，中值滤波+限幅去尖峰
输出：E:\Processed\{name}_processed_30Hz.txt

5.4 align_data.py
输入：
  - E:\Data\Video\*.avi
  - E:\Data\labels\*.npy
  - E:\Data\Record\*_completed_summary.csv
  - E:\Data\BreathBelt\*_processed_30Hz.txt
逻辑：
  1. 正则匹配同一实验的所有文件，解析时间戳
  2. 从 CSV 读取 prep 开始和 end 结束时间
  3. 计算视频帧索引范围（视频帧率 = 5 Hz）
  4. 截取呼吸带信号中对应实验时段，并计算每一帧内的平均幅度
输出：
  - E:\Data\Aligned\{subj}_{sess}_{mode}_aligned.npz
    包含：frame_idx, label, amplitude, breath_idx
  - 汇总表 sessions_summary.csv

5.5 visualize_labels.py
用法：python visualize_labels.py <video_path> <label_npy_path>
功能：播放视频，顶部叠加当前帧号和标签文字，支持暂停/调速

5.6 belt_data_plot_waveforms.py
输入：E:\Processed\*_processed_30Hz.txt
输出：三类波形图（单文件、按被试、按模式）保存至 E:\Processed\Figures

5.7 generate_training_samples.py
输入：aligned .npz + 处理后呼吸带 .txt
逻辑：
  - 滑窗参数：窗长 30 秒（150 帧），步长 5 秒（25 帧）
  - eupnea 模式：仅在标签全为 1 的连续段中滑动
  - apnea 模式：在包含屏气段的连续段中滑动，窗口必须完整包含至少一个 apnea 段
  - 对应提取 900 点呼吸带信号（30 Hz × 30 秒）
输出：
  - samples_metadata.csv（包含 subject, session, mode, 起始帧, apnea 区间等）
  - samples_data.npz（video_labels: N×150，breath_signals: N×900）

5.8 split_dataset.py
输入：samples_metadata.csv + samples_data.npz
方法：按 subject_id 分层随机划分，比例 80/10/10，随机种子 42
输出：
  - train/val/test_metadata.csv
  - train/val/test_data.npz（格式同 samples_data.npz）

5.9 thermal_breath_dataset.py
自定义 torch.utils.data.Dataset
读取视频片段（从对应视频中按帧索引读取）、标签和呼吸信号
视频帧转为灰度单通道，输出张量形状 (1, 150, H, W)

5.10 train.py
示例训练脚本，使用 3D 卷积 → 全连接回归呼吸波形（MSE 损失）
使用者需根据实际任务修改网络结构

5.11 工具脚本
  - npy_to_csv.py：将标签 .npy 转为 (frame_index, label) CSV
  - npz_to_csv.py：将 aligned .npz 转为 (frame_idx, label, amplitude, breath_idx) CSV


6. 样本构建策略

窗口定义：时长 30 秒（150 帧 @5 fps），滑窗步长 5 秒（25 帧）
样本筛选：
  - 正常呼吸（eupnea）：窗口内所有帧标签均为 1
  - 屏气（apnea_x）：窗口内必须完整覆盖至少一个 apnea_hold 段（标签 2），且窗口不超出标签为 1 或 2 的连续区域
呼吸带信号：截取窗口起始帧对应的 breath_idx，取 900 个采样点（30 Hz × 30 s），不足则补 NaN
每条样本记录元数据：被试、会话、模式、窗口起始帧、apnea 起止帧


7. 输出数据格式

7.1 对齐数据（aligned .npz）
  - frame_idx: (N,) uint32，帧索引
  - label: (N,) uint8，标签 0/1/2
  - amplitude: (N,) float32，该帧内平均呼吸带幅度（可能为 NaN）
  - breath_idx: (N,) int32，该帧起始对应的呼吸带采样点索引（30 Hz）

7.2 样本数据（samples_data.npz 及划分后文件）
  - video_labels: (M, 150) uint8，每窗口的帧级标签序列
  - breath_signals: (M, 900) float32，对应的呼吸带波形片段

7.3 元数据 CSV 列
  subject_id, session_id, mode, start_frame, end_frame, apnea_start, apnea_end


8. 关键参数

  视频帧率：5 fps（优先读取实际帧率），用于标签生成和对齐
  呼吸带目标采样率：30 Hz
  呼吸带中间重采样频率：200 Hz
  带通滤波：0.1 – 15 Hz，4 阶 Butterworth
  去尖峰：中值滤波核大小 5，限幅阈值 4 倍标准差
  滑窗时长：30 秒（150 帧 / 900 呼吸点）
  滑窗步长：5 秒（25 帧）
  数据集划分比例：80/10/10（按被试），随机种子 42


9. 注意事项

  - 时间对齐误差：视频与呼吸带的时间戳均取自文件名，若设备时钟未同步，可能引入恒定偏移，建议人工校准。
  - CSV 标注质量：脚本强依赖 prep 和 end 阶段的行，缺失将导致对齐失败；apnea_hold 段的标注需与实际屏气动作一致。
  - 信号噪声：原始呼吸带可能包含运动伪迹或电气噪声，可调整滤波和限幅参数；过度滤波会导致呼吸波形细节丢失。
  - 视频预处理：伽马值需根据相机类型微调，以保证面部区域对比度。
  - 所有脚本中的路径（E:\...）为示例，实际使用时请修改为对应数据目录。
  - 缺失数据：滑窗时若呼吸带采样点不足，会补 NaN，训练时需考虑掩蔽或插值处理。
  - 数据集划分严格按被试独立进行，避免信息泄露。


10. 快速开始

1. （可选）预处理热红外视频：
   python process_thermal_video.py   （修改脚本内的 INPUT_DIR/OUTPUT_DIR）
2. 生成帧级标签：
   python batch_label_with_log.py    （确保 VIDEO_DIR 下 avi 和 csv 配对）
3. 处理呼吸带原始信号：
   python belt_data_process.py       （修改 INPUT_DIR/OUTPUT_DIR）
4. 多模态时间对齐：
   python align_data.py              （确认各子目录路径）
5. （可选）可视化检查标签：
   python visualize_labels.py <video.avi> <labels.npy>
6. 构建训练样本：
   python generate_training_samples.py
7. 划分训练/验证/测试集：
   python split_dataset.py
8. 训练模型：
   python train.py                   （需自行定义模型或修改示例中的 SimpleVideoRegressor）

各脚本内部路径需根据实际数据目录调整，建议先在小样本上验证全部流程，确认无误后再批量处理。