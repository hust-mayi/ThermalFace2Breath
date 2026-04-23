import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------- 配置 -----------------------------
INPUT_DIR = r"E:\RawVideo\labels"      # npy文件所在目录
OUTPUT_DIR = r"E:\RawVideo\labels_csv" # csv输出目录（可修改，若为None则与npy同目录）
# ---------------------------------------------------------------

def convert_npy_to_csv(input_dir: str, output_dir: str = None):
    """
    将指定目录下的所有 .npy 文件转换为 .csv 文件
    csv 包含两列: frame_index, label
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"目录不存在: {input_dir}")
        return
    
    # 输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    npy_files = list(input_path.glob("*.npy"))
    if not npy_files:
        print(f"未找到任何 .npy 文件: {input_dir}")
        return
    
    print(f"找到 {len(npy_files)} 个 .npy 文件")
    
    for npy_path in npy_files:
        try:
            labels = np.load(npy_path)
            # 创建 DataFrame
            df = pd.DataFrame({
                "frame_index": np.arange(len(labels)),
                "label": labels
            })
            # 输出 csv 文件名与 npy 相同，扩展名改为 .csv
            csv_filename = npy_path.stem + ".csv"
            csv_filepath = output_path / csv_filename
            df.to_csv(csv_filepath, index=False)
            print(f"转换成功: {npy_path.name} -> {csv_filepath}")
        except Exception as e:
            print(f"转换失败 {npy_path.name}: {e}")

if __name__ == "__main__":
    convert_npy_to_csv(INPUT_DIR, OUTPUT_DIR)