import os
import re
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# ----------------------------- 配置 -----------------------------
VIDEO_DIR = r"E:\RawVideo"               # 视频所在目录
CSV_SUFFIX = "_completed_summary.csv"    # CSV文件后缀
OUTPUT_DIR = r"E:\RawVideo\labels"       # 标签输出目录（None表示与视频同目录）
FPS = 5.0                                # 视频帧率（已知5Hz，脚本会优先使用实际帧率）
LOG_DIR = None                           # 日志目录，None表示在VIDEO_DIR下创建logs文件夹
LOG_LEVEL = logging.INFO                 # 日志级别
# ---------------------------------------------------------------

def setup_logger(log_dir=None, log_level=logging.INFO):
    """配置日志：同时输出到控制台和文件"""
    logger = logging.getLogger('VideoLabeling')
    logger.setLevel(log_level)
    
    if logger.handlers:
        return logger
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_dir is None:
        log_dir = Path(VIDEO_DIR) / "logs"
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_filename = log_dir / f"labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"日志文件已创建: {log_filename}")
    return logger

logger = setup_logger(LOG_DIR, LOG_LEVEL)

def parse_video_start_from_filename(filename: str) -> datetime:
    """
    从视频文件名提取开始时间，格式如 '20260413-151631412' 或 '_20260413_152244'
    返回 datetime 对象
    """
    # 匹配 8位日期 + 分隔符 + 6位时间 + 可选毫秒（3位或6位）
    pattern = r'(\d{8})[-_](\d{6})(\d{3})?'
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"无法从文件名解析时间: {filename}")
    date_str = match.group(1)          # YYYYMMDD
    time_str = match.group(2)          # HHMMSS
    ms_str = match.group(3) if match.group(3) else '000'
    dt_str = f"{date_str} {time_str}.{ms_str}"
    return datetime.strptime(dt_str, "%Y%m%d %H%M%S.%f")

def extract_subject_session_from_filename(filename: str):
    """
    从文件名提取 subject_id, session_id, base_mode（不含日期后缀）
    示例:
        subject_401_session_301_eupnea_20260413-151631412.avi -> (401, 301, 'eupnea')
        subject_403_session_107_apnea_20_20260413_203148838.avi -> (403, 107, 'apnea_20')
    返回 (subject, session, base_mode) 或 (None, None, None)
    """
    # 匹配模式：subject_数字_session_数字_模式名（模式名可能含下划线），直到遇到 _数字 或 -数字 或文件扩展名
    pattern = r'subject_(\d+)_session_(\d+)_([a-zA-Z]+(?:_[a-zA-Z0-9]+)*?)(?=[_-]\d{8}|\.|$)'
    match = re.search(pattern, filename)
    if not match:
        # 尝试更宽松的匹配：允许模式名后直接跟日期（无分隔符）
        pattern2 = r'subject_(\d+)_session_(\d+)_([a-zA-Z]+(?:_[a-zA-Z0-9]+)*?)\d{8}'
        match = re.search(pattern2, filename)
    if not match:
        return None, None, None
    subject = int(match.group(1))
    session = int(match.group(2))
    base_mode = match.group(3)
    return subject, session, base_mode

def find_matching_csv(video_path: str, csv_files: list):
    """
    根据视频文件中的 subject, session, base_mode 匹配对应的 CSV 文件
    """
    video_name = os.path.basename(video_path)
    subj, sess, mode = extract_subject_session_from_filename(video_name)
    if subj is None:
        logger.warning(f"视频文件名无法解析subject/session: {video_name}")
        return None
    for csv_path in csv_files:
        csv_name = os.path.basename(csv_path)
        csv_subj, csv_sess, csv_mode = extract_subject_session_from_filename(csv_name)
        if csv_subj == subj and csv_sess == sess and csv_mode == mode:
            return csv_path
    logger.warning(f"未找到匹配的CSV: subject={subj}, session={sess}, mode={mode} (视频: {video_name})")
    return None

def load_experiment_intervals(csv_path: str):
    """
    从 CSV 中提取实验起止时间和各阶段区间
    返回 (exp_start, exp_end, intervals_list)
    intervals_list: 每个元素为 (start, end, category)
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 第一行是总体信息，从中提取实验起止时间
    exp_start = pd.to_datetime(df.iloc[0]['experiment_start_time'])
    exp_end = pd.to_datetime(df.iloc[0]['experiment_end_time'])
    
    # 筛选包含阶段信息的行（phase_name 非空）
    phase_rows = df[df['phase_name'].notna()].copy()
    intervals = []
    for _, row in phase_rows.iterrows():
        start = pd.to_datetime(row['actual_start_time'])
        end = pd.to_datetime(row['actual_end_time'])
        category = row['category']
        intervals.append((start, end, category))
    logger.debug(f"从 {os.path.basename(csv_path)} 加载了 {len(intervals)} 个阶段")
    return exp_start, exp_end, intervals

def generate_frame_labels(video_path: str, csv_path: str, fps: float = FPS) -> np.ndarray:
    """
    为单个视频生成标签数组
    标签: 0=实验外, 1=实验内正常呼吸, 2=呼吸暂停(apnea_hold)
    """
    # 获取视频信息
    video_start = parse_video_start_from_filename(os.path.basename(video_path))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # 确定帧间隔
    if actual_fps > 0 and abs(actual_fps - fps) > 0.1:
        logger.info(f"视频 {os.path.basename(video_path)} 实际帧率 {actual_fps:.2f} Hz，使用实际帧率")
        frame_interval = timedelta(seconds=1/actual_fps)
    else:
        frame_interval = timedelta(seconds=1/fps)
    
    # 加载实验阶段
    exp_start, exp_end, intervals = load_experiment_intervals(csv_path)
    
    # 生成标签
    labels = np.zeros(total_frames, dtype=np.int8)
    for i in range(total_frames):
        frame_time = video_start + i * frame_interval
        if frame_time < exp_start or frame_time > exp_end:
            labels[i] = 0
            continue
        # 查找所属阶段
        for start, end, cat in intervals:
            if start <= frame_time <= end:
                labels[i] = 2 if cat == 'apnea_hold' else 1
                break
        else:
            # 理论上不应发生，若发生则视为正常呼吸
            logger.warning(f"帧 {i} 时间 {frame_time} 落在实验区间但无匹配阶段，设为标签1")
            labels[i] = 1
    return labels

def batch_process(video_dir: str, output_dir: str = None):
    """
    批量处理目录下所有 .avi 视频，生成对应标签文件
    """
    video_dir = Path(video_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"标签输出目录: {output_dir}")
    else:
        logger.info("标签将保存在视频同目录下")
    
    # 收集文件
    video_files = list(video_dir.glob("*.avi"))
    csv_files = list(video_dir.glob(f"*{CSV_SUFFIX}"))
    logger.info(f"找到 {len(video_files)} 个视频文件，{len(csv_files)} 个CSV文件")
    
    if not video_files:
        logger.warning("未找到任何 .avi 文件，处理结束")
        return
    
    success_count = 0
    fail_count = 0
    
    for video_path in video_files:
        logger.info(f"开始处理视频: {video_path.name}")
        csv_path = find_matching_csv(str(video_path), csv_files)
        if csv_path is None:
            logger.error(f"跳过 {video_path.name}: 无匹配CSV")
            fail_count += 1
            continue
        
        logger.info(f"匹配到CSV: {os.path.basename(csv_path)}")
        try:
            labels = generate_frame_labels(str(video_path), csv_path)
            # 确定输出路径
            if output_dir:
                out_path = output_dir / f"{video_path.stem}_labels.npy"
            else:
                out_path = video_path.parent / f"{video_path.stem}_labels.npy"
            np.save(out_path, labels)
            logger.info(f"成功生成标签: 总帧数={len(labels)}, "
                        f"标签0={np.sum(labels==0)}, 标签1={np.sum(labels==1)}, 标签2={np.sum(labels==2)}")
            logger.info(f"标签文件保存至: {out_path}")
            success_count += 1
        except Exception as e:
            logger.exception(f"处理视频 {video_path.name} 时发生异常: {e}")
            fail_count += 1
    
    logger.info(f"批量处理完成: 成功 {success_count} 个，失败 {fail_count} 个")

if __name__ == "__main__":
    batch_process(VIDEO_DIR, OUTPUT_DIR)