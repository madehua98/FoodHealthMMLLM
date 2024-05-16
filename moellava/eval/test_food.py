import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from PIL import Image
import math


def split_jsonl(input_file, n):
    """
    Split a JSONL file into n smaller JSONL files.
    
    :param input_file: Path to the input JSONL file.
    :param n: Number of parts to split the file into.
    """
    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    # Ensure the output directory exists
    output_dir = os.path.dirname(input_file)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Calculate the number of lines per part
    total_lines = len(lines)
    lines_per_part = total_lines // n
    remainder = total_lines % n
    
    start = 0
    for i in range(n):
        end = start + lines_per_part + (1 if i < remainder else 0)
        part_lines = lines[start:end]
        
        # Write the part to a new file
        part_file = os.path.join(output_dir, f'{base_filename}_part{i+1}.jsonl')
        with open(part_file, 'w', encoding='utf-8') as pf:
            pf.writelines(part_lines)
        
        start = end



def read_json(file):
    with open(file, mode='r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_jsonl(file):
    data_list = []
    with open(file, mode='r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data = json.loads(line)
            data_list.append(data)
    return data_list

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=64):
    #assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_name = "MoE-LLaVA-Phi2-2.7B-4e"
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    

    if args.return_gating_logit is not None:
        from moellava.utils import get_gating_logit_by_hook
        # print(model)
        # state_dict = model.state_dict()
        # parameters_to_save = {}
        # for name, param in state_dict.items():
        #     if "mm_projector.image_spatial_proj" in name:
        #         parameters_to_save[name] = param.cpu().detach()
        #     torch.save(parameters_to_save, "/media/LLM_data/model/moellava/checkpoints/llavaqwen1.8B_mm_projector.bin")
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}
    image_processor = processor['image']
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    batch_size = 1
    
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # Load questions and create DataLoader with DistributedSampler
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, batch_size=1)
    sampler = DistributedSampler(data_loader.dataset, num_replicas=torch.distributed.get_world_size(), rank=local_rank)
    data_loader = DataLoader(data_loader.dataset, sampler=sampler, batch_size=1)
    
    # Check if answers file exists
    if os.path.exists(args.answers_file):
        answer_file = read_jsonl(args.answers_file)
    else:
        answer_file = None
    
    cnt = -1
    questions_loaders = zip(data_loader, questions)
    for (input_ids, image_tensor), line in tqdm(questions_loaders, total=len(questions)):
        cnt += 1
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        # Skip if the question is already processed
        if answer_file is not None and any(item["question_id"] == idx for item in answer_file):
            print(f"Data {idx} already validated, skipping")
            continue
        
        input_ids = input_ids.to(device=local_rank, non_blocking=True)
        
        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=local_rank, non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True if args.return_gating_logit is None else False
            )
        
        # Decode and save output
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        with open(args.answers_file, mode="a+", encoding='utf-8') as fw:
            fw.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {}
            }) + "\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)                                                                                                                                           
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--return_gating_logit", type=str, default="true")
    args = parser.parse_args()

    eval_model(args)
    
    
    
    ###############计算指标
    # gold_answer_file_path = "/mnt/data_llm/json_file/101_answers.jsonl"
    # answer_file_path = "/home/data_llm/madehua/FoodHealthMMLLM/eval/food101/answers/MoE-LLaVA-Phi2-2.7B-4e-v101_0426.jsonl"
    # from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    # from nltk.metrics.distance import edit_distance
    # accuracy, precision, recall, f1 = get_metrics(answer_file_path, gold_answer_file_path)
    # print(accuracy, precision, recall, f1)

