import requests
import json
import sys
from utils.file_utils import *

idx = 8000 + int(sys.argv[1])
json_id = int(sys.argv[1])

url = f'http://127.0.0.1:{idx}/v1/chat/completions'
json_file  = "/media/fast_data/mid_json/%d.json" % json_id
save_file  = "/media/fast_data/mid_json/%d_txt_filtered.json" % json_id
info = load_json(json_file)

res = []
for item in tqdm(info):
    conv = item['conversations']
    instruct = ''
    for line in conv:
        instruct += line['value']
    instruct = instruct.replace('\n', '').replace('<image>', '')
    instruct = "Please judge the following paragraph contains food or not. '" + instruct[:800] + "' If related to food, please answer yes, otherwise no。"

    myobj = {
        "model": "Qwen/Qwen1.5-7B-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruct},
        ],
        # "do_sample": True,
        # "repetition_penalty": 2.0,
        # "frequency_penalty": 0.3,
        # "top_p": 0,
        # "n": 1,
        # "max_tokens": 2048,
        "stream": False
    }
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(myobj), headers=headers)
        # print(response)
        # print(myobj, '\n', response.json()["choices"][0]["message"]["content"].strip())
        ans = response.json()["choices"][0]["message"]["content"].strip()
        if 'no' in ans.lower():
            continue
        res.append(item)
    except Exception as e:
        pass
print(save_file, len(res))
save_json(save_file, res)

'''
nohup python -u scripts/txt_filter_food.py 0 > query0.out &
nohup python -u scripts/txt_filter_food.py 1 > query1.out  &
nohup python -u scripts/txt_filter_food.py 2 > query2.out  &
nohup python -u scripts/txt_filter_food.py 3 > query3.out  &
nohup python -u scripts/txt_filter_food.py 4 > query4.out  &
nohup python -u scripts/txt_filter_food.py 6 > query6.out  &
nohup python -u scripts/txt_filter_food.py 7 > query7.out  &
nohup python -u scripts/txt_filter_food.py 8 > query8.out  &
nohup python -u scripts/txt_filter_food.py 9 > query9.out  &
'''
'''
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8000 > log0.out &
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8001 > log1.out &
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8002 > log2.out &
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8003 > log3.out &
CUDA_VISIBLE_DEVICES=4 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8004 > log4.out &
CUDA_VISIBLE_DEVICES=6 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8006 > log6.out &
CUDA_VISIBLE_DEVICES=7 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8007 > log7.out &
CUDA_VISIBLE_DEVICES=8 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8008 > log8.out &
CUDA_VISIBLE_DEVICES=9 nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype half --max-model-len 1024 --port 8009 > log9.out &

'''