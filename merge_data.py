import json
import os
import random

def read_json(file):
    with open(file, mode='r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def write_json(file, object):
    with open(file, mode='w', encoding='utf-8') as fw:
        json.dump(fp=fw, obj=object, ensure_ascii=False, indent=2)


def load_json_files(list):
    """读取指定目录下的所有 JSON 文件并返回一个包含所有条目的列表"""
    json_entries = []
    # 遍历目录中的所有文件
    for file_path in list:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(len(data))
            json_entries.extend(data)  # 假设每个文件都是一个列表
    return json_entries

def shuffle_and_reassign_ids(json_entries):
    """打乱列表并重新分配 ID"""
    random.shuffle(json_entries)  # 打乱顺序
    for index, entry in enumerate(json_entries):
        entry['id'] = str(index)  # 更新 ID
    return json_entries

def save_to_json(json_entries, output_file):
    """将合并后的 JSON 数据保存到文件"""
    with open(output_file, 'w') as f:
        print(len(json_entries))
        json.dump(json_entries, f, indent=2)


json_folder="/mnt/data_llm/json_file"
"""
${json_folder}/101_train_prompt10.json
${json_folder}/2k_train_prompt10.json
${json_folder}/172_train_prompt10.json
${json_folder}/172_ingredient_train_prompt10.json
"""
train_101 = f"{json_folder}/101_train_prompt10.json"
train_2k = f"{json_folder}/2k_train_prompt10.json"
train_172 = f"{json_folder}/172_train_prompt10.json"
train_172_ingredient = f"{json_folder}/172_ingredient_train_prompt10.json"
train_0429 = f"{json_folder}/food_recognition_0429.json"
train_list = [train_101, train_2k, train_172, train_172_ingredient]

# entries = load_json_files(train_list)
# shuffled_entries = shuffle_and_reassign_ids(entries)
# save_to_json(shuffled_entries, train_0429)

food_recognition_0429 = read_json(train_0429)
for i in food_recognition_0429:
    if "food-101" in i["image"]:
        print(i["image"])