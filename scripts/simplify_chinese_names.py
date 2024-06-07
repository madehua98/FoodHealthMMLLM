from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil
import torch

# 确认GPU可用
if not torch.cuda.is_available():
    raise EnvironmentError("This model requires a GPU, but none is available.")

device = torch.device("cuda:7")

# 检查字符是否为中文
def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

# 检查字符串是否包含中文
def contains_chinese(s):
    return any(is_chinese(char) for char in s)


# 初始化模型和tokenizer
model_name = "Qwen/Qwen1.5-14B-Chat-AWQ"
model = AutoModelForCausalLM.from_pretrained(model_name
                                             ).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_response(folder_name):
    # 简化prompt，让模型直接返回新的文件夹名
    # prompt = f"文件夹‘{folder_name}’应去除所有品牌名和描述性词汇，仅保留核心名词。直接回答新文件夹名字。"
    # prompt = f"请精炼食品名称中的品牌名和与食物无关的描述性词汇，并直接给出答案。例如:\n'(散)苹果'应回答'苹果'。\n'干桂圆'应回答'干桂圆'。\n'喜之郎精品果冻系列'应回答'果冻'。\n现在请精炼名称：'{folder_name}'。\n"
    prompt = f"请通过去掉与食物类型无关的描述性词汇（如散称、散称系列、优质、十九怪、新式以及标点符号数字）并纠正错别字，来抽取食物类型关键词。若为散称且未说明具体食品类型则回答散装食品。例如:\n'(散)苹果'应回答苹果\n'干桂圆'应回答干桂圆\n'喜之郎精品果冻系列'应回答果冻\n'开腹鲤鱼'应回答鲤鱼\n'散称混沌，烧麦'应回答馄炖烧麦\n'散称12元散点'应回答散装食品\n那么，'{folder_name}'应回答"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs['input_ids'], max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "应回答" in response:
        response = response.split("应回答")[-1]
    # 直接从响应文本中提取处理后的文件夹名
    processed_name = response.strip().replace("。", "").replace("：", "").replace("'", "").replace("、", "").replace("（", "").replace("）", "").replace("，", "").replace(" ", "")

    return processed_name



def merge_folders(source, destination):
    for item in os.listdir(source):
        src_path = os.path.join(source, item)
        dst_path = os.path.join(destination, item)
        if os.path.isdir(src_path):
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            merge_folders(src_path, dst_path)
        else:
            if not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)
            else:
                print(f"Skipping {src_path} as {dst_path} already exists.")

def process_folder_names(directory):
    count = 0
    invalid_count = 0
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path) and contains_chinese(folder_name):
            processed_name = get_response(folder_name)
            if '无答案' in processed_name or len(processed_name) > len(folder_name): # invalid
                invalid_count += 1
                continue
            print(folder_name, processed_name, "\n")

            # new_folder_path = os.path.join(directory, processed_name)
            # if not os.path.exists(new_folder_path):
            #     shutil.move(folder_path, new_folder_path)
            #     print(f"Renamed '{folder_name}' to '{processed_name}'")
            # else:
            #     if folder_path != new_folder_path:
            #         merge_folders(folder_path, new_folder_path)
            #         shutil.rmtree(folder_path)
            #         print(f"Merged '{folder_name}' into '{processed_name}'")

        count += 1
        if count > 200:
            break
    print(count, invalid_count)

directory = "/media/fast_data/new_fresh_devices"
process_folder_names(directory)

'''
pip install autoawq
CUDA_VISIBLE_DEVICE=5,6,7 python -u scripts/simplify_chinese_names.py
'''