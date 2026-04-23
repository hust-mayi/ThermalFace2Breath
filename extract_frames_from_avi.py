import cv2
import os
import glob

def extract_frames_from_avi(input_dir, output_dir=None, frame_interval=1):
    """
    从目录中的所有 .avi 视频提取帧并保存为图片。

    参数:
        input_dir (str): 存放 .avi 视频的目录。
        output_dir (str): 保存帧图像的根目录，若为 None 则自动在 input_dir 同级创建。
        frame_interval (int): 每隔多少帧保存一帧，默认为 1（保存每一帧）。
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 '{input_dir}' 不存在。")
        return

    # 设置输出根目录
    if output_dir is None:
        output_dir = input_dir.rstrip(os.sep) + "_frames"
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 .avi 文件（不区分大小写）
    video_paths = glob.glob(os.path.join(input_dir, "*.avi")) + \
                   glob.glob(os.path.join(input_dir, "*.AVI"))

    if not video_paths:
        print(f"在 '{input_dir}' 中未找到任何 .avi 文件。")
        return

    print(f"找到 {len(video_paths)} 个视频文件，开始提取帧...")

    for video_path in video_paths:
        # 获取视频文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # 为该视频创建子输出目录
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件：{video_path}，跳过。")
            continue

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔保存帧
            if frame_count % frame_interval == 0:
                out_filename = f"frame_{saved_count:06d}.jpg"
                out_path = os.path.join(video_output_dir, out_filename)
                cv2.imwrite(out_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"完成：{video_name} -> 共 {frame_count} 帧，实际保存 {saved_count} 张图片。")

    print("所有视频处理完毕。")

if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = r"E:\ProcessedVideo\test"          # 视频所在目录
    OUTPUT_DIR = None                          # 输出根目录（None 则自动生成）
    FRAME_INTERVAL = 1                         # 每隔1帧保存一帧（即全部保存）

    extract_frames_from_avi(INPUT_DIR, OUTPUT_DIR, FRAME_INTERVAL)