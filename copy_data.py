import shutil
import threading
import time
import os
from queue import Queue
import json
from tqdm import tqdm


# 文件复制函数
def copy_file(src, dest, queue):
    os.makedirs(os.path.dirname(dest), exist_ok=True)  # 确保目标目录存在
    shutil.copy2(src, dest)
    queue.put(src)  # 通知队列此文件已完成复制

# 线程工作函数
def worker(file_queue, src_base, dest_base, progress):
    while not file_queue.empty():
        src_file = file_queue.get()
        rel_path = os.path.relpath(src_file, src_base)
        dest_file = os.path.join(dest_base, rel_path)
        copy_file(src_file, dest_file, progress)
        file_queue.task_done()

# 主函数
def main(src_folder, dest_folder, max_threads=500):
    start_time = time.time()

    # 遍历源文件夹中的文件和子目录
    files_to_copy = []
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            files_to_copy.append(os.path.join(root, file))
    total_files = len(files_to_copy)
    # 创建队列
    file_queue = Queue()
    progress_queue = Queue()

    # 将文件添加到队列
    for file in files_to_copy:
        file_queue.put(file)
    # 创建并启动线程
    threads = []
    for _ in range(min(max_threads, total_files)):
        thread = threading.Thread(target=worker, args=(file_queue, src_folder, dest_folder, progress_queue))
        thread.start()
        threads.append(thread)
    # 显示进度
    copied_files = 0
    with tqdm(total=total_files, desc="Copying files") as pbar:
        while copied_files < total_files:
            progress_queue.get()
            copied_files += 1
            pbar.update(1)  # 更新进度条

    # 等待所有线程完成
    file_queue.join()
    for thread in threads:
        thread.join()

    end_time = time.time()
    print(f"All files copied in {end_time - start_time} seconds.")
    print("*"*100)
# 使用示例
main('/media/fast_data/datacomp_small', '/mnt/data_llm/datacomp_small')
# main("/media/fast_data/recipe1M", "/mnt/data_llm/food_images/recipe1M")
# main("/media/fast_data/food-101", "/mnt/data_llm/food_images/food-101")
# main("/media/fast_data/VireoFood172", "/mnt/data_llm/food_images/VireoFood172")
# main("/media/fast_data/nutrition5k_dataset", "/mnt/data_llm/food_images/nutrition5k_dataset")
# main("/media/fast_data/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b", "/mnt/data_llm/food_images/moe_data")
def change_image_path(source_path, dest_path, replace_path):
    new_datas = []
    with open(source_path, mode="r", encoding="utf-8") as f:
        datas = json.load(f)
        for data in datas:
            data["image"] = data["image"].replace(replace_path, "/mnt/data_llm/food_images/moe_data")
            new_datas.append(data)
    with open(dest_path, mode="w", encoding="utf-8") as f:
        json.dump(obj=new_datas, fp=f, ensure_ascii=False, indent=4)

old_json_path = '/media/fast_data/json_file/'
new_json_path = '/mnt/data_llm/json_file/'
file_names = [
    # "101_train_prompt1.json",
    # "101_test_prompt1.json",
    # "172_ingredient_test_prompt1.json",
    # "172_ingredient_train_prompt1.json",
    # "172_ingredient_val_prompt1.json",
    # "172_test_prompt1.json",
    # "172_train_prompt1.json",
    # "172_val_prompt1.json",
    # "2k_train_prompt1.json",
    # "2k_test_prompt1.json",
    # "nutrition5k_test.json",
    # "nutrition5k_train.json"
    # "train_ingredient_QA.json",
    # "train_recipe_QA.json",
    # "train_title_QA.json",
    "mix_food.json"
]
# replace_path = "/media/fast_data/huggingface/hub/datasets--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b"
# for file_name in file_names:
#     change_image_path(old_json_path+file_name, new_json_path+file_name, replace_path)