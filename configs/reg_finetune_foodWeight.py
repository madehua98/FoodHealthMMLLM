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


inp_size = 224  # (w,h)
config_file = '/home/xuzhenbo/food_dataset/weights_samples.pkl'  #
num_classes = 1

precision = 16
gpu = 4
args = dict(

    cuda=True,
    display=False,
    display_it=5,
    save=True,
    re_train=True,
    opt_param='two_lr',  # ['e2e', 'fix_bn', 'fc_only', 'two_lr]
    opt_type='adam',  # ['adam', 'radam', 'rmsprop', 'sgd]
    # loss options
    loss_type='SmoothCE',
    loss_opts={},

    save_dir='./cls_foodWeight',
    resume_path=os.path.join(project_root, 'cls_foodWeight/cls_foodWeight/best-acc1=3.8003-epoch=009-max120.ckpt'),
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
        'name': 'common_reg',  # multiple datasource
        'kwargs': {
            'config_file': config_file,
            'type': 'train',
            'im_size': inp_size,
            'size': 128000,
        },
        'batch_size': 128,
        'workers': 12,
        # 'workers': 0,
    },

    val_dataset={
        'name': 'common_reg',
        'kwargs': {
            'config_file': config_file,
            'type': 'val',
            'im_size': inp_size,
            # 'size': 12800,
        },
        'batch_size': 128,
        'workers': 12,
        # 'workers': 0,
    },

    model={
        'name': 'common_reg_net',
        'kwargs': {
            'num_classes': num_classes,
            'net_type': 'faster_vit',
            'resolution': [inp_size, inp_size],
        }
    },
    lr=5e-4,  # wo center
    n_epochs=120,
    val_check_interval=1.0,
    start_epoch=1,
    max_disparity=192.0,
)


def get_args():
    return copy.deepcopy(args)

'''
cd /home/xuzhenbo/MoE-LLaVA/cls_foodWeight && conda activate trt_ascend
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6 nohup python -u ../train2.py reg_finetune_foodWeight > weight_4card.out & 


GHMR + 5e-5
####### diff_avg: 0.4911, val seed: 0.08880624, val loss: 0.08880624

MSE + 5e-4
CUDA_VISIBLE_DEVICES=4,6,8,9 nohup python -u ../train2.py reg_finetune_foodWeight > weight_4card.out & 
####### diff_avg: 0.3238, val seed: 0.25441225, val loss: 0.25441225
####### diff_avg: 0.2872, val seed: 0.20992990, val loss: 0.20992990
####### diff_avg: 0.2631, val seed: 0.18070006, val loss: 0.18070006

MSE + 5e-4 + OEEM
CUDA_VISIBLE_DEVICES=4,6,8,9 nohup python -u ../train2.py reg_finetune_foodWeight > weight_4card_oeem.out & 
####### diff_avg: 0.4315, val seed: 0.01292398, val loss: 0.01292398

GHMR + 5e-4
CUDA_VISIBLE_DEVICES=4,6,8,9 nohup python -u ../train2.py reg_finetune_foodWeight > weight_4card_ghmr.out & 
####### diff_avg: 0.2760, val seed: 0.04398897, val loss: 0.04398897
'''