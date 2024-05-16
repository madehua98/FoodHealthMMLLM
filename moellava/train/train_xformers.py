# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
from moellava.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)
import os
replace_llama_attn_with_xformers_attn()

from moellava.train.train import train

if __name__ == "__main__":
    os.environ['NCCL_TIMEOUT'] = '3600000'
    train()
