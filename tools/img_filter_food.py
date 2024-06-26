import os.path
import sys

from utils.file_utils import *
# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 sys.path 中
sys.path.append(project_root)

# root = "/media/fast_data/mid_json"
# json_files = make_dataset(root, suffix="_txt_filtered.json")
# device_count = 9
# all_items = []
# for json_file in tqdm(json_files):
#     all_items += load_json(json_file)
# random.shuffle(all_items)
# print('shuffle finished', len(all_items))

# dst_file = "/media/fast_data/mid_json/%d_before_img_filtered.json"
# idxs = [0,1,2,3,4,6,7,8,9]
# split_count = len(idxs)
# gap = len(all_items) // split_count
# for i, idx in enumerate(idxs):
#     dst_file_ = dst_file % idx
#     print(i, dst_file_)
#     save_json(dst_file_, all_items[i*gap:i*gap+gap])

# #filtering
# from PIL import Image
# import torchvision.transforms as transforms
# from datasets_own.common_cls_dataset import SquarePad
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# inp_size = (384, 384)
# transform = transforms.Compose([
#     SquarePad(),
#     transforms.Resize(inp_size, interpolation=3),  # BICUBIC interpolation
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
# ])
# jit_model_path = '/media/fast_data/tool_models/has_food_fv_gpu.pt'
# device = torch.device("cuda")
# model = torch.jit.load(jit_model_path).to(device)

# json_id = 1
# json_file  = "/media/fast_data/mid_json/%d_before_img_filtered.json" % json_id
# save_file  = "/media/fast_data/mid_json/%d_txt_img_filtered.json" % json_id
# info = load_json(json_file)

# res = []
# for item in tqdm(info):
#     real_input_path = item["image"]
#     # try:
#     real_input = transform(Image.open(real_input_path).convert('RGB')).unsqueeze(0).to(device)
#     print(real_input.shape)
#     re = model(real_input)
#     re = torch.softmax(re, dim=-1)
#     hasFood = re[0, 1].item() > 0.3
#     # print(real_input_path, re, hasFood)
#     if hasFood:
#         res.append(item)
#     # except Exception as e:
#     #     pass
# print(save_file, len(res), len(res)/len(info))
# save_json(save_file, res)

'''
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/img_filter_food.py 0 > log0.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/img_filter_food.py 1 > log1.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u tools/img_filter_food.py 2 > log2.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u img_filter_food.py 3 > log3.out &
CUDA_VISIBLE_DEVICES=4 nohup python -u tools/img_filter_food.py 4 > log4.out &
CUDA_VISIBLE_DEVICES=6 nohup python -u tools/img_filter_food.py 6 > log6.out &
CUDA_VISIBLE_DEVICES=7 nohup python -u tools/img_filter_food.py 7 > log7.out &
CUDA_VISIBLE_DEVICES=8 nohup python -u tools/img_filter_food.py 8 > log8.out &
CUDA_VISIBLE_DEVICES=9 nohup python -u img_filter_food.py > log9.out &

'''




import json
import sys
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from datasets_own.common_cls_dataset import SquarePad
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import tarfile
import random
from faster_vit import faster_vit_1_224, faster_vit_2_224

import torch.nn as nn

# Define transformations
inp_size = (384, 384)
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(inp_size, interpolation=3),  # BICUBIC interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

os.environ['CUDA_VISIBLE_DEVICES'] = f'8'

directory_path_set = '/media/fast_data/datacomp_1b/extracted_shards'
num_gpus = torch.cuda.device_count()
print(num_gpus)
# Load model

# 假设您的模型定义如下
model = faster_vit_2_224(pretrained=None)  # 这里设置为 None，因为我们要加载训练好的参数
model.head = nn.Linear(model.num_features, 2)

model_load_path = '/media/fast_data/tool_models/hasfood_fastervit_2_224_1.pth'
model.load_state_dict(torch.load(model_load_path))

# 将模型转换为 TorchScript 格式
scripted_model = torch.jit.script(model)
# 保存模型
jit_model_path = '/media/fast_data/tool_models/hasfood_fastervit_2_224_1.pth'
scripted_model.save(jit_model_path)

device = torch.device("cuda")
model = torch.jit.load(jit_model_path).to(device)
model.eval()
print(model)



random.seed(0)
directory_paths = [os.path.join(directory_path_set, f) for f in os.listdir(directory_path_set)]
directory_paths = random.sample(directory_paths, 5)
# Parameters for batching
batch_size = 512
res = []
batch_inputs = []
batch_items = []
food_images_status = {}

def save_jsonl(filename, object):
    with open(filename, 'a+') as fw:
        fw.write(json.dumps(object) + '\n')

def load_jsonl(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            res.append(json.loads(line))
    return res

threshold_lower = 0.6
threshold_upper = 1.0

# Iterate over directories
for directory_path in directory_paths:
    real_input_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('jpg')]
    food_images_status = {}
    for real_input_path in tqdm(real_input_paths):
        hasFood_count = 0
        try:
            # Load and transform image
            real_input = transform(Image.open(real_input_path).convert('RGB')).unsqueeze(0).to(device)
            input_shape = real_input.shape
            batch_inputs.append(real_input)
            batch_items.append(real_input_path)

            # If batch is full, process it
            if len(batch_inputs) >= batch_size:
                batch_tensor = torch.cat(batch_inputs).to(device)
                with torch.no_grad():
                    re = model(batch_tensor)
                    re = torch.softmax(re, dim=-1)
                    #hasFood = re[:, 1] > threshold_lower
                    hasFood = (re[:, 1] > threshold_lower) & (re[:, 1] < threshold_upper)
                    res.extend([batch_items[i] for i in range(len(hasFood)) if hasFood[i]])
                    for i in range(len(hasFood)):
                        if hasFood[i]:
                            food_images_status[batch_items[i]] = 1
                        else:
                            food_images_status[batch_items[i]] = 0

                # Clear batch
                batch_inputs = []
                batch_items = []

        except Exception as e:
            print(f"Error processing {real_input_path}: {e}")

    # Process remaining images if any
    if batch_inputs:
        batch_tensor = torch.cat(batch_inputs).to(device)
        with torch.no_grad():
            re = model(batch_tensor)
            re = torch.softmax(re, dim=-1)
            hasFood = (re[:, 1] > threshold_lower) & (re[:, 1] < threshold_upper)
            res.extend([batch_items[i] for i in range(len(hasFood)) if hasFood[i]])
            for i in range(len(hasFood)):
                if hasFood[i]:
                    food_images_status[batch_items[i]] = 1
                else:
                    food_images_status[batch_items[i]] = 0

        # Clear batch after processing
        batch_inputs = []
        batch_items = []
    has_food_count = 0
    no_food_count = 0
    for key, value in food_images_status.items():
        if value == 1:
            has_food_count += 1
        elif value == 0:
            no_food_count += 1
    filename = f'/media/fast_data/datacomp_1b/threshold_{threshold_lower}_{threshold_upper}.jsonl'
    print(len(food_images_status))
    save_jsonl(filename, food_images_status)
#Save results
# print(save_file, len(res), len(res)/len(info))
# save_json(save_file, res)

# root = "/media/fast_data/mid_json"
# json_files = make_dataset(root, suffix="_txt_img_filtered.json")
# all_items = []
# for json_file in tqdm(json_files):
#     all_items += load_json(json_file)
# random.shuffle(all_items)
# print('shuffle finished', len(all_items))
# for el in all_items[:10]:
#     print(el)
# save_json(os.path.join(root, 'food_final.json'), all_items)



# """
# 将判断不为食品的图像下载到本地
# """

filename_2 = '/media/fast_data/datacomp_1b/threshold_0.0_0.1.jsonl'
filename_1 = '/media/fast_data/datacomp_1b/threshold_0.1_0.2.jsonl'
filename_3 = '/media/fast_data/datacomp_1b/threshold_0.2_0.3.jsonl'
filename_4 = '/media/fast_data/datacomp_1b/threshold_0.3_0.4.jsonl'
filename_5 = '/media/fast_data/datacomp_1b/threshold_0.4_0.5.jsonl'

source_dir = '/media/fast_data/datacomp_1b/extracted_shards'


res_1 = load_jsonl(filename_1)
res_2 = load_jsonl(filename_2)
res_3 = load_jsonl(filename_3)
res_4 = load_jsonl(filename_4)
res_5 = load_jsonl(filename_5)

res_1_list = []
count = 0
count_total = 0
tasks = []
tasks_pos = []
# for res1 in res_3:
#     destination_dir_1 = f'/media/fast_data/datacomp_1b/res_2_3'
#     #destination_dir_2 = f'/media/fast_data/datacomp_1b/res_pos_5'
#     for k,v in res1.items():
#         if v == 1:
#             file_name = os.path.basename(k)
#             destination_file = os.path.join(destination_dir_1, file_name)
#             tasks.append([k, destination_file])
#             count += 1
#         # if v == 1:
#         #     file_name = os.path.basename(k)
#         #     destination_file = os.path.join(destination_dir_2, file_name)
#         #     tasks_pos.append([k, destination_file])
#         #     count += 1
#     count_total += len(res1)

# random.seed(0)
# image_dir = f'/media/fast_data/datacomp_1b/res_2_3'
# image_dir1 = f'/media/fast_data/datacomp_1b/res_2_3_part1'
# image_dir2 = f'/media/fast_data/datacomp_1b/res_2_3_part2'
# images_path = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
# random.shuffle(images_path)
# images_path1 = images_path[:6000]
# images_path2 = images_path[6000:12000]
# tasks1 = [[os.path.join(image_dir, f), os.path.join(image_dir1, f)] for f in images_path1]
# tasks2 = [[os.path.join(image_dir, f), os.path.join(image_dir2, f)] for f in images_path2]
# print(tasks1)
# for task in tqdm(tasks1,total=len(tasks1)):
#     shutil.copy(task[0], task[1])

# for task in tqdm(tasks2, total=len(tasks2)):
#     shutil.copy(task[0], task[1])


# print(count)
# print(count_total)

# res_2_list = []
# for res2 in res_2:
#     for k,v in res2.items():
#         if v == 0:
#             res_2_list.append(k)

# res_3_list = []
# for res3 in res_3:
#     for k,v in res3.items():
#         if v == 0:
#             res_3_list.append(k)

# res_4_list = []
# for res4 in res_4:
#     for k,v in res4.items():
#         if v == 0:
#             res_4_list.append(k)
