"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
'''
use scripts/generate_config_for_hasFood.py to create config
'''
import copy
import os
from PIL import Image
import torch
from config import *


inp_size = 384  # (w,h)
config_file = '/home/xuzhenbo/food_dataset/hasFood_samples.pkl'  #
num_classes = 2

precision = 16
gpu = 4
args = dict(

    cuda=True,
    display=False,
    display_it=5,
    save=True,
    re_train=True,
    opt_param='two_lr',  # ['e2e', 'fix_bn', 'fc_only', 'two_lr]
    opt_type='sgd',  # ['adam', 'radam', 'rmsprop', 'sgd]
    # loss options
    loss_type='SmoothCE',
    loss_opts={},

    save_dir='./cls_hasFood',
    resume_path=os.path.join(project_root, 'cls_hasFood/cls_hasFood/best-acc1=0.9727-epoch=059-max060.ckpt'),
    # remove_key_words=['poincare_head.'],
    gpus=gpu,
    check_val_every_n_epoch=5,
    train1=False,
    train2=True,
    final_test=True,
    precision=precision,
    strategy='ddp_find_unused_parameters_false',
    gradient_clip_val=1.0,

    train_dataset={
        'name': 'common_cls',  # multiple datasource
        'kwargs': {
            'config_file': config_file,
            'type': 'train',
            'im_size': inp_size,
            'size': 32000,
        },
        'batch_size': 64,
        'workers': 12,
        # 'workers': 0,
    },

    val_dataset={
        'name': 'common_cls',
        'kwargs': {
            'config_file': config_file,
            'type': 'val',
            'im_size': inp_size,
        },
        'batch_size': 64,
        'workers': 8,
        # 'workers': 0,
    },

    model={
        'name': 'common_cls_net',
        'kwargs': {
            'num_classes': num_classes,
            'net_type': 'faster_vit',
            'resolution': [inp_size, inp_size],
        }
    },
    lr=5e-5,  # wo center
    n_epochs=60,
    val_check_interval=1.0,
    start_epoch=1,
    max_disparity=192.0,
)


def get_args():
    return copy.deepcopy(args)

'''
cd /home/xuzhenbo/MoE-LLaVA/cls_hasFood
CUDA_VISIBLE_DEVICES=4,6,8,9 nohup python -u ../train2.py cls_finetune_hasFood > fv2_4card.out & 
'''