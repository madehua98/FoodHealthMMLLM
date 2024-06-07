'''
如果边缘高度达到了最大高度的80%，则认为这个图片存在大面积在外的可能性，则需要去掉

'''
import cv2

# from tools.compute_fg_mask import *
# import json
# import sys
# def load_json(filename):
#     return json.load(open(filename, 'r'))
#
#
# output_train_file = '/media/fast_data/food_weight_predict_code/weight_dataset_train2.json'
# output_test_file = '/media/fast_data/food_weight_predict_code/weight_dataset_test2.json'
# items = load_json(output_train_file) + load_json(output_test_file)
# # key2Ind = {info['image']:ind for ind, info in enumerate(infos)}
# infos = [el['image'] for el in items]
#
# assert len(sys.argv) > 1
# total_num, index = int(sys.argv[1]), int(sys.argv[2])
# gap = len(infos) // total_num + 1
# paths = infos[index * gap:(index + 1) * gap]
# # t = Test(PGNet, paths, '/mnt/Downloads/0model-150--loss=0.8678293228149414.pth')
# PGNet = torch.jit.load('/mnt/Downloads/PGNet_front_20231214_gpu.pt')
# t = Test(PGNet, paths, None)
# t.save()

'''
conda activate trt_ascend && export PYTHONPATH=/home/xuzhenbo/FoodHealthMMLLM
CUDA_VISIBLE_DEVICES=4 nohup python -u scripts/compute_weight_dataset_fg_mask.py 5 0 > temp_4.out &
CUDA_VISIBLE_DEVICES=6 nohup python -u scripts/compute_weight_dataset_fg_mask.py 5 1 > temp_6.out &
CUDA_VISIBLE_DEVICES=7 nohup python -u scripts/compute_weight_dataset_fg_mask.py 5 2 > temp_7.out &
CUDA_VISIBLE_DEVICES=8 nohup python -u scripts/compute_weight_dataset_fg_mask.py 5 3 > temp_8.out &
CUDA_VISIBLE_DEVICES=9 nohup python -u scripts/compute_weight_dataset_fg_mask.py 5 4 > temp_9.out &
'''

# write a func to judge a image is valid or not
from utils.file_utils import make_dataset
import numpy as np

image_root = "/mnt/data_llm/new_fresh_devices2"
npz_files = make_dataset(image_root, suffix='.npz')

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def process_npz_file(npz_file_path):
    mask = np.load(npz_file_path)['mask'][0]
    mask = (sigmoid(mask) > 0.3).astype(np.float32) * 255

    image_path = npz_file_path.replace('.npz', '.jpg')
    image = cv2.imread(image_path)
    mask_resize = cv2.resize(np.repeat(mask[:,:,np.newaxis], 3, -1), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    vis = np.concatenate([image, mask_resize], axis=0)
    cv2.imwrite('/mnt/Downloads/1.jpg', vis)
    b=1

for npz_file in npz_files:
    process_npz_file(npz_file)
b=1