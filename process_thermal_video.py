import cv2
import os
import glob
import numpy as np

def adjust_gamma(image, gamma=0.5):
    """
    使用查找表(LUT)对图像进行伽马校正以提高处理速度。
    gamma < 1.0: 增强亮度 (变亮)
    gamma > 1.0: 降低亮度 (变暗)
    """
    # 建立查找表，将[0, 255]的值映射到调整后的伽马值
    invGamma = 1.0 / gamma
    # 公式: O = (I / 255)^(gamma) * 255
    table = np.array([((i / 255.0) ** gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # 使用 cv2.LUT 应用查找表
    return cv2.LUT(image, table)

def process_videos(input_dir, output_dir, gamma_value=0.5):
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 获取所有 avi 文件
    search_pattern = os.path.join(input_dir, '*.avi')
    video_files = glob.glob(search_pattern)

    if not video_files:
        print(f"在 {input_dir} 下没有找到 .avi 视频文件。")
        return

    print(f"共找到 {len(video_files)} 个视频，开始处理...")

    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, filename)
        
        # 打开视频流
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {filename}")
            continue

        # 获取原视频的属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置视频编码器 (XVID 是一种常用的 avi 编码格式)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # 创建 VideoWriter 对象，注意 isColor=False 表示写入灰度视频
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break # 视频读取完毕

            # 1. 转换为灰度图
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. 反转灰度图像 (获取负片)
            # 也可以用 255 - gray_frame 实现
            inverted_frame = cv2.bitwise_not(gray_frame)

            # 3. 应用伽马校正增强亮度
            final_frame = adjust_gamma(inverted_frame, gamma=gamma_value)

            # 写入处理后的帧
            out.write(final_frame)
            frame_count += 1

        # 释放资源
        cap.release()
        out.release()
        
        print(f"[{i+1}/{len(video_files)}] 处理完成: {filename} (共 {frame_count}/{total_frames} 帧)")

    print("所有视频处理完毕！")

if __name__ == "__main__":
    # 定义输入和输出路径
    INPUT_DIR = 'E:/RawVideo'
    OUTPUT_DIR = 'E:/ProcessedVideo'
    
    # 定义伽马值 (需要增强亮度，所以 gamma 设置为小于 1 的值，比如 0.4 到 0.6)
    # 你可以根据实际热红外视频的效果调整这个值
    GAMMA_VALUE = 0.5 

    process_videos(INPUT_DIR, OUTPUT_DIR, gamma_value=GAMMA_VALUE)