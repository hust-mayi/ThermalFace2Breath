import cv2
import numpy as np
import sys
from pathlib import Path

# 标签映射
LABEL_NAMES = {
    0: "Before/After Experiment",
    1: "Normal Breathing",
    2: "Apnea Hold"
}
LABEL_COLORS = {
    0: (0, 255, 255),   # 黄色
    1: (0, 255, 0),     # 绿色
    2: (0, 0, 255)      # 红色
}

def visualize(video_path: str, label_path: str, fps_scale: float = 1.0):
    """
    播放视频并叠加标签
    video_path: 视频文件路径
    label_path: 对应的 .npy 标签文件路径
    fps_scale: 播放速度缩放（1.0=原速，>1加快，<1减慢）
    """
    # 加载标签
    labels = np.load(label_path)
    print(f"标签文件加载成功，总帧数: {len(labels)}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 5.0  # 根据实际情况调整
    delay_ms = int(1000 / (original_fps * fps_scale))
    
    print(f"视频总帧数: {total_frames}, 原始帧率: {original_fps:.2f} fps, 播放延迟: {delay_ms} ms")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break
        
        # 获取当前帧标签（如果标签数量少于视频帧数，只用到 min）
        if frame_idx < len(labels):
            label = labels[frame_idx]
        else:
            label = -1  # 未知
        
        # 在图像上添加标签文本和背景
        text = LABEL_NAMES.get(label, f"Unknown ({label})")
        color = LABEL_COLORS.get(label, (128, 128, 128))
        
        # 绘制半透明背景条
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # 绘制文字
        cv2.putText(frame, f"Frame: {frame_idx}   Label: {text}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 显示帧号
        cv2.imshow("Label Verification", frame)
        
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            break
        elif key == ord(' '):  # 空格暂停/继续
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord(' '):
                    break
                elif k == ord('q') or k == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python visualize_labels.py <视频文件路径> <标签文件路径>")
        print("示例: python visualize_labels.py E:/RawVideo/subject_403_session_107_apnea_20_20260413-203148838.avi E:/RawVideo/labels/subject_403_session_107_apnea_20_20260413-203148838_labels.npy")
        sys.exit(1)
    
    video_path = sys.argv[1]
    label_path = sys.argv[2]
    
    # 可选：调整播放速度，例如0.5倍速
    visualize(video_path, label_path, fps_scale=0.8)