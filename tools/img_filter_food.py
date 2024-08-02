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
nohup python -u img_filter_food.py 1 0.0 1.0 0.1 1 > 1.out &
nohup python -u img_filter_food.py 2 0.0 1.0 0.1 2 > 2.out &
nohup python -u img_filter_food.py 3 0.0 1.0 0.1 3 > 3.out &
nohup python -u img_filter_food.py 4 0.0 1.0 0.1 4 > 4.out &
nohup python -u img_filter_food.py 5 0.0 1.0 0.1 5 > 5.out &
nohup python -u img_filter_food.py 6 0.0 1.0 0.1 6 > 6.out &
nohup python -u img_filter_food.py 7 0.0 1.0 0.1 7 > 7.out &
nohup python -u img_filter_food.py 0 0.0 1.0 0.1 0 > 0.out &
'''


import torch
print(torch.cuda.is_available())

import json
import sys
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from datasets_own.common_cls_dataset import SquarePad
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
from faster_vit import faster_vit_1_224, faster_vit_2_224
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Define transformations
inp_size = (384, 384)
# transform = transforms.Compose([
#     SquarePad(),
#     transforms.Resize(inp_size, interpolation=3),  # BICUBIC interpolation
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
# ])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
num_gpus = torch.cuda.device_count()
print(num_gpus)

random.seed(0)

def save_jsonl(filename, object):
    with open(filename, 'a+') as fw:
        fw.write(json.dumps(object) + '\n')

def load_jsonl(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            res.append(json.loads(line))
    return res

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except (IOError, OSError):
            return self.__getitem__((idx + 1) % len(self.image_paths))  # Skip this image and get the next one
        
        if self.transform:
            image = self.transform(image)
        return image, image_path

def main(device_id, threshold_lower, threshold_upper, interval_step, idx):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    inp_size = (384, 384)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    directory_path_set = '/ML-A100/team/mm/models/laion2b/extracted_shards3'
    directory_paths = [os.path.join(directory_path_set, f) for f in os.listdir(directory_path_set)]
    
    directory_paths.sort()
    # 将目录分成8份
    num_partitions = 8
    part_size = len(directory_paths) // num_partitions
    partitioned_directories = [directory_paths[i*part_size : (i+1)*part_size] for i in range(num_partitions)]
    partitioned_directories_new = partitioned_directories[idx]

    model = faster_vit_2_224(pretrained=None)
    model.head = nn.Linear(model.num_features, 2)
    model_load_path = '/ML-A100/team/mm/models/hasfood_fastervit_2_224_v2.pth'
    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.head.load_state_dict(checkpoint['head_state_dict'])
    model = model.to(device)
    model.eval()

    batch_size = 512
    for directory_path in tqdm(partitioned_directories_new, total=len(partitioned_directories_new)):
        real_input_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('jpg')]
        dataset = ImageDataset(real_input_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        food_images_status = {}
        for batch_inputs, batch_items in dataloader:
            batch_inputs = batch_inputs.to(device)
            with torch.no_grad():
                outputs = model(batch_inputs)
                probabilities = torch.softmax(outputs, dim=-1)
                food_probabilities = probabilities[:, 1]
                for i, item in enumerate(batch_items):
                    if food_probabilities[i] < threshold_lower:
                        interval_id = 0  # 低于下限
                    elif food_probabilities[i] >= threshold_upper:
                        interval_id = num_intervals  # 高于上限
                    else:
                        interval_id = int((food_probabilities[i] - threshold_lower) // interval_step) + 1
                    food_images_status[item] = interval_id

        filename = f'/ML-A100/team/mm/models/laion2b/threshold_record.jsonl'
        if food_images_status != {}:
            save_jsonl(filename, food_images_status)
        
import sys
device_id = int(sys.argv[1])
threshold_lower = float(sys.argv[2])
threshold_upper = float(sys.argv[3])
interval_step = float(sys.argv[4])
idx = int(sys.argv[5])
print(idx)
main(device_id, threshold_lower, threshold_upper, interval_step, idx)
print("done")
    
    
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

# filename_2 = '/ML-A100/team/mm/models/datacomp_1b/threshold_0.0_0.1.jsonl'
# filename_1 = '/ML-A100/team/mm/models/datacomp_1b/threshold_0.9_1.0.jsonl'
# filename_3 = '/ML-A100/team/mm/models/datacomp_1b/threshold_0.2_0.3.jsonl'
# filename_4 = '/ML-A100/team/mm/models/datacomp_1b/threshold_0.3_0.4.jsonl'
# filename_5 = '/ML-A100/team/mm/models/datacomp_1b/threshold_0.4_0.5.jsonl'

# source_dir = '/ML-A100/team/mm/models/datacomp_1b/extracted_shards'


# res_1 = load_jsonl(filename_1)
# res_2 = load_jsonl(filename_2)
# res_3 = load_jsonl(filename_3)
# res_4 = load_jsonl(filename_4)
# res_5 = load_jsonl(filename_5)

# res_1_list = []
# count = 0
# count_total = 0
# tasks = []
# tasks_pos = []
# for res1 in res_1:
#     destination_dir_1 = f'/ML-A100/team/mm/models/datacomp_1b/res/9_0'
#     os.makedirs(destination_dir_1, exist_ok=True)
#     #destination_dir_2 = f'/media/fast_data/datacomp_1b/res_pos_5'
#     for k,v in res1.items():
#         if v == 1:
#             file_name = os.path.basename(k)
#             destination_file = os.path.join(destination_dir_1, file_name)
#             tasks.append([k, destination_file])
#             count += 1
#     count_total += len(res1)


# for task in tqdm(tasks,total=len(tasks)):
#     shutil.copy(task[0], task[1])
    
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



# for task in tqdm(tasks2, total=len(tasks2)):
#     shutil.copy(task[0], task[1])

