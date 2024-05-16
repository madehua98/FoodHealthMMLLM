from huggingface_hub import snapshot_download
import os
import json

# os.environ['https_proxy'] = '127.0.0.1:7890'
# os.environ['http_proxy'] = '127.0.0.1:7890'
# snapshot_download(
#   repo_id="Qwen/Qwen-1_8B",
#   cache_dir="/media/LLM_data/model",
#   proxies={"https": "http://localhost:7890"},
# )
# snapshot_download(
#   repo_id="openai/clip-vit-large-patch14-336",
#   cache_dir="/media/LLM_data/model",
#   proxies={"https": "http://localhost:7890"},
# )

################################################################ 模型下载

from huggingface_hub import snapshot_download, login

os.environ["HF_ENDPOINT"]="https://huggingface.co/"
#login("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")
repo_id = "Qwen/Qwen1.5-7B-Chat"
local_dir = "/media/fast_data/model/Qwen1.5-7B-Chat"
snapshot_download(
  repo_id=repo_id,
  local_dir=local_dir,
  resume_download=False,
  max_workers=1
)



################################################################# llama3-8b模型推理测试

# import transformers
# import torch

# os.environ['CUDA_VISIBLE_DEVICES']='5'
# os.environ['HF_DATASETS_OFFLINE']='1'
# model_id = "/media/fast_data/model/Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="auto",
# )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True
# )

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# print(outputs[0]["generated_text"][len(prompt):])





