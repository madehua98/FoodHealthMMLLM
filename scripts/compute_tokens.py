import transformers
from moellava.train.train import make_supervised_data_module
from moellava.model.language_model.llava_phi import LlavaPhiForCausalLM
import torch
from moellava.train.train import LazySupervisedDataset
import json

CACHE_FOLDER="/media/fast_data/huggingface/hub/"
model_name_or_path = "microsoft/phi-2"
# image_tower = "openai/clip-vit-large-patch14-336"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    model_max_length=4096,
    cache_dir=CACHE_FOLDER,
    padding_side="right",
    use_fast=False,
)
tokenizer.add_special_tokens({'unk_token': '<|extra_0|>'})
tokenizer.pad_token = tokenizer.unk_token


# model = LlavaPhiForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
#     # attn_implementation="flash_attention_2",
#     # torch_dtype=torch.bfloat16,
#     **bnb_model_from_pretrained_args
# )
# model.to(torch.bfloat16)

# model.get_model().initialize_vision_modules(
#     model_args=model_args,
#     fsdp=training_args.fsdp
# )
# image_tower = model.get_image_tower()
# image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

# data_args.image_processor = image_tower.image_processor
# data_args.is_multimodal = True

# model.config.image_aspect_ratio = data_args.image_aspect_ratio
# model.config.tokenizer_padding_side = tokenizer.padding_side



# data_module = make_supervised_data_module(tokenizer=tokenizer,
#                                             data_args=data_args)

# trainer = LLaVATrainer(model=model,
#                 tokenizer=tokenizer,
#                 args=training_args,
#                 **data_module)




def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tokenizer(string)
    num_tokens = len(encoding['input_ids'])
    return num_tokens

def read_json(file_name):
    with open(file=file_name, mode='r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


def get_ratio_exceed_threshold(filename, threshold):
    data = read_json(filename)
    long_list = []
    for i in range(len(data)):
        conv = ''
        for conversation in data[i]["conversations"]:
            conv += conversation['value']
        #print(conv)
        token_num = num_tokens_from_string(conv)
        if token_num > 512:
            long_list.append(token_num)

    ratio = len(long_list) /len(data)
    return ratio


json_folder="/mnt/data_llm/json_file/"
# --data_path ${json_folder}/101_train_prompt1.json \
#             ${json_folder}/train_ingredient_QA.json ${json_folder}/train_recipe_QA.json ${json_folder}/train_title_QA.json \
#             ${json_folder}/2k_train_prompt1.json \
#             ${json_folder}/172_train_prompt1.json ${json_folder}/172_ingredient_train_prompt1.json \
#             ${json_folder}/nutrition5k_train.json ${json_folder}/mix_food.json\
#             ${json_folder}/weight_dataset_train2.json ${json_folder}/train_nutrition_QA.json \

# 定义文件名列表
file_names = [
    '101_train_prompt1.json',
    'train_ingredient_QA.json',
    'train_recipe_QA.json',
    'train_title_QA.json',
    '2k_train_prompt1.json',
    '172_train_prompt1.json',
    '172_ingredient_train_prompt1.json',
    'nutrition5k_train.json',
    'mix_food.json',
    'weight_dataset_train2.json',
    'train_nutrition_QA.json'
]

# 假设 json_folder 是包含文件的路径
json_folder="/mnt/data_llm/json_file/"

# 遍历每个文件名
for file_name in file_names:
    # 拼接完整的文件路径
    full_path = json_folder + file_name
    threshold = 512
    try:
        ratio = get_ratio_exceed_threshold(full_path, threshold)
        print(f"{full_path}: 超过{threshold}令牌的比例为 {ratio}")
    except Exception as e:
        print(f"处理文件 {full_path} 时发生错误: {e}")