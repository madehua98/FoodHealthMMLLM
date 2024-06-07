# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
from moellava.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

#replace_llama_attn_with_xformers_attn()

from moellava.train.train import train
import os
os.environ['PATH'] += '/home/xuzhenbo/anaconda3/envs/moellava/bin/'

if __name__ == "__main__":
    train()
