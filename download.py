from huggingface_hub import snapshot_download
import os
import json

os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ['http_proxy'] = '127.0.0.1:7890'
snapshot_download(
  repo_id="Qwen/Qwen-1_8B",
  cache_dir="/media/LLM_data/model",
  proxies={"https": "http://localhost:7890"},
)
snapshot_download(
  repo_id="openai/clip-vit-large-patch14-336",
  cache_dir="/media/LLM_data/model",
  proxies={"https": "http://localhost:7890"},
)

dict_template = {"id": "none", "image": "image_path", "conversations":[]}


