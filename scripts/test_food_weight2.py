import torch
import json
from PIL import Image
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from copy import deepcopy

def try_parse_float(value):
    """尝试将字符串转换为浮点数，如果失败则返回None"""
    try:
        return float(value)
    except ValueError:
        print(f"Warning: Unable to convert '{value}' to float.")
        return None

def main():
    disable_torch_init()
    json_file_path = '/mnt/data_llm/json_file/weight_dataset_test_0426.json'
    model_path = '/mnt/data_llm/model/checkpoints/llavaphi-2.7b-finetune-moe-weight_0426'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = 'MoE-LLaVA-Phi2-2.7B-4e'
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, False, False, device=device)
    image_processor = processor['image']

    errors = []
    total_true_weight = 0
    num_samples = 0

    with open(json_file_path, 'r') as f, open('parse_errors.txt', 'w') as log_file:
        data = json.load(f)
        for item in data:
            conv_template = conv_templates["phi"].copy()
            #conv_template = deepcopy(conv_templates["phi"].copy())
            image_path = item['image']
            value_string = item['conversations'][1]['value'].split()[0]
            true_weight = try_parse_float(value_string)
            if true_weight is None:
                error_info = {
                    "id": item.get("id", "unknown"),
                    "image": image_path,
                    "value": value_string,
                    "conversations": item['conversations']
                }
                log_file.write(json.dumps(error_info) + '\n')
                continue

            total_true_weight += true_weight
            num_samples += 1

            image_tensor = image_processor.preprocess(Image.open(image_path).convert('RGB'), return_tensors='pt')['pixel_values'].to(device, dtype=torch.float16)
            inp = item['conversations'][0]['value']
            conv_template.append_message(conv_template.roles[0], inp)
            conv_template.append_message(conv_template.roles[1], None)
            prompt = conv_template.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            stop_str = conv_template.sep if conv_template.sep_style != SeparatorStyle.TWO else conv_template.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                predicted_weight = try_parse_float(outputs.split()[0])
                if predicted_weight is None:
                    continue

                error = abs(predicted_weight - true_weight)
                errors.append(error)

    mean_true_weight = total_true_weight / num_samples if num_samples > 0 else 0
    mae = sum(errors) / len(errors) if errors else 0
    mae_percent = (mae / mean_true_weight) * 100 if mean_true_weight > 0 else 0

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Error as a Percent of Mean: {mae_percent}%")

if __name__ == '__main__':
    main()