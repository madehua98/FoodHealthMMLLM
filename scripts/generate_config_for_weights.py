import os.path

from utils.file_utils import *

image_root = '/home/data_llm/new_fresh_devices2'

all_samples0 = make_dataset(image_root, suffix='kg.jpg')
train_info, val_info = {}, {}
print(len(all_samples0))
random.shuffle(all_samples0)

def process(im_path):
    weight_ = float(os.path.basename(im_path).split('-')[-2].split('_')[-1])
    try:
        Image.open(im_path)
    except:
        return None
    if weight_ > 6:
        return None
    return im_path


res = run_imap_multiprocessing(process, all_samples0, 32)
all_samples = [el for el in res if el is not None]
print(len(all_samples))
weights = [float(os.path.basename(el).split('-')[-2].split('_')[-1]) for el in all_samples]
train_len = int(len(all_samples) * 0.9)
train_info = {k:v for k,v in zip(all_samples[:train_len], weights[:train_len])}
val_info = {k:v for k,v in zip(all_samples[train_len:], weights[train_len:])}
print(len(train_info), len(val_info))
save_pickle('/home/xuzhenbo/food_dataset/weights_samples.pkl', {'train':train_info, 'val':val_info})
'''
2140355
100%|███████████████████████████████| 2140355/2140355 [09:30<00:00, 3751.71it/s]
1895182
1705663 189519
'''

