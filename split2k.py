import json
import os
import random
from  tqdm import tqdm


def find_image_file(directory_path):
    jpg_files = []
    for file in os.listdir(directory_path):
        if file.endswith(".jpg"):
            jpg_files.append(file)
    return jpg_files

def write_json_file(data, json_file_path):
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def write_txt_file(data, file_path):
    with open(file_path, 'a') as file:
        # 写入文本
        file.write(data+"\n")
def split_2k(image_folder_path):
    split_rate = 0.9
    output_path_train = "/media/fast_data/Data/Food2k_complete/train.txt"
    output_path_test = "/media/fast_data/Data/Food2k_complete/test.txt"
    total_split_image_files_train = []
    total_split_image_files_test = []
    for i in tqdm(range(2000)):
        image_files = find_image_file(image_folder_path+f"/{i}")
        split_image_files_train = random.sample(image_files, int(len(image_files) * split_rate))
        split_image_files_train_set = set(split_image_files_train)
        all_image_files_set = set(image_files)
        split_image_files_test_set = all_image_files_set - split_image_files_train_set
        split_image_files_test = list(split_image_files_test_set)
        new_split_image_files_train = []
        for image_file in split_image_files_train:
            new_image_file = f"/{i}/"+image_file
            new_split_image_files_train.append(new_image_file)
        total_split_image_files_train.append(new_split_image_files_train)
        new_split_image_files_test = []
        for image_file in split_image_files_test:
            new_image_file = f"/{i}/"+image_file
            new_split_image_files_test.append(new_image_file)
        total_split_image_files_test.append(new_split_image_files_test)
    for files_train in tqdm(total_split_image_files_train):
        for image_file_train in files_train:
            write_txt_file(image_file_train, output_path_train)
    for files_test in tqdm(total_split_image_files_test):
        for image_file_test in files_test:
            write_txt_file(image_file_test, output_path_test)
image_folder_path = "/media/fast_data/Data/Food2k_complete"

# split_2k(image_folder_path)


for i in range(10):
    print(i)
    i += 1

