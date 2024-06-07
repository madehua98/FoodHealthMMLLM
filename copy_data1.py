import os
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import shutil

def copy_item(src_path, dest_dir):
    """
    使用shutil.copy2进行文件复制或shutil.copytree进行目录复制
    :param src_path: 源路径（文件或目录）
    :param dest_dir: 目标目录路径
    """
    try:
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)
        print(f"Copy from {src_path} to {dest_path} completed successfully.")
    except Exception as e:
        print(f"Error during copy from {src_path} to {dest_path}: {e}")

def main():
    # 定义源目录和目标目录
    src_dir = "/mnt/data_llm/model/checkpoints/checkpoints-phi-2.7b-moe-v2k_0426"
    dest_dir = "/media/fast_data/model/checkpoints/checkpoints-phi-2.7b-moe-v2k_0426"
    
    # 获取源目录下的所有文件和目录
    src_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    
    # 使用tqdm显示进度条
    copy_item_with_dest = partial(copy_item, dest_dir=dest_dir)
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(copy_item_with_dest, src_path) for src_path in src_paths]
        for _ in tqdm(futures, total=len(futures)):
            pass

if __name__ == "__main__":
    main()
