import os.path

from utils.file_utils import *
# root = "/media/fast_data/mid_json"
# json_files = make_dataset(root, suffix="_txt_filtered.json")
# device_count = 9
# all_items = []
# for json_file in tqdm(json_files):
#     all_items += load_json(json_file)
# random.shuffle(all_items)
# print('shuffle finished', len(all_items))
#
# dst_file = "/media/fast_data/mid_json/%d_before_img_filtered.json"
# idxs = [0,1,2,3,4,6,7,8,9]
# split_count = len(idxs)
# gap = len(all_items) // split_count
# for i, idx in enumerate(idxs):
#     dst_file_ = dst_file % idx
#     print(i, dst_file_)
#     save_json(dst_file_, all_items[i*gap:i*gap+gap])

# # filtering
# from PIL import Image
# import torchvision.transforms as transforms
# from datasets.common_cls_dataset import SquarePad
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
# 
# idx = 8000 + int(sys.argv[1])
# json_id = int(sys.argv[1])
# json_file  = "/media/fast_data/mid_json/%d_before_img_filtered.json" % json_id
# save_file  = "/media/fast_data/mid_json/%d_txt_img_filtered.json" % json_id
# info = load_json(json_file)
# 
# res = []
# for item in tqdm(info):
#     real_input_path = item["image"]
#     try:
#         real_input = transform(Image.open(real_input_path).convert('RGB')).unsqueeze(0).to(device)
#         re = model(real_input)
#         re = torch.softmax(re, dim=-1)
#         hasFood = re[0, 1].item() > 0.3
#         # print(real_input_path, re, hasFood)
#         if hasFood:
#             res.append(item)
#     except Exception as e:
#         pass
# print(save_file, len(res), len(res)/len(info))
# save_json(save_file, res)

'''
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/img_filter_food.py 0 > log0.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u tools/img_filter_food.py 1 > log1.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u tools/img_filter_food.py 2 > log2.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u tools/img_filter_food.py 3 > log3.out &
CUDA_VISIBLE_DEVICES=4 nohup python -u tools/img_filter_food.py 4 > log4.out &
CUDA_VISIBLE_DEVICES=6 nohup python -u tools/img_filter_food.py 6 > log6.out &
CUDA_VISIBLE_DEVICES=7 nohup python -u tools/img_filter_food.py 7 > log7.out &
CUDA_VISIBLE_DEVICES=8 nohup python -u tools/img_filter_food.py 8 > log8.out &
CUDA_VISIBLE_DEVICES=9 nohup python -u tools/img_filter_food.py 9 > log9.out &

'''

root = "/media/fast_data/mid_json"
json_files = make_dataset(root, suffix="_txt_img_filtered.json")
all_items = []
for json_file in tqdm(json_files):
    all_items += load_json(json_file)
random.shuffle(all_items)
print('shuffle finished', len(all_items))
for el in all_items[:10]:
    print(el)
save_json(os.path.join(root, 'food_final.json'), all_items)