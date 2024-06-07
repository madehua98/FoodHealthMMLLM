import torch
import json
from PIL import Image
import re
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

def parse_nutritional_info(nutritional_string, log_file, item):
    try:
        nutrients = nutritional_string.split(', ')
        nutrient_dict = {}
        for nutrient in nutrients:
            key, value = nutrient.split(': ')
            match = re.match(r"[-+]?[0-9]*\.?[0-9]+", value)
            if match:
                clean_value = match.group(0)
                nutrient_dict[key] = float(clean_value)
            else:
                log_file.write(f"解析营养成分值失败 {key} 来自条目 {item.get('id', 'unknown')}: {json.dumps(item)}\n")
                return None
        return nutrient_dict
    except Exception as e:
        log_file.write(f"解析条目出错 {item.get('id', 'unknown')}: {json.dumps(item)} 错误: {str(e)}\n")
        return None

def main():
    disable_torch_init()
    json_file_path = '/mnt/data_llm/json_file/nutrition5k_test_modified1.json'
    model_path = '/mnt/data_llm/model/checkpoints/llavaphi-2.7b-finetune-moe-v0426'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_4bit, load_8bit = False, False
    model_name = 'MoE-LLaVA-Phi2-2.7B-4e'
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    image_processor = processor['image']
    conv_mode = "phi"
    totals = {key: 0 for key in ['Calories', 'Mass', 'Fat', 'Carbs', 'Protein']}
    counts = {key: 0 for key in totals.keys()}
    errors = {key: [] for key in totals.keys()}

    with open(json_file_path, 'r') as f, open('failed_conversations_nutrition.txt', 'w') as log_file:
        data = json.load(f)
        for item in data:
            conv_template = conv_templates["phi"].copy()
            image_path = item['image']
            try:
                with Image.open(image_path) as img:
                    image_tensor = image_processor.preprocess(img.convert('RGB'), return_tensors='pt')['pixel_values'].to(device, dtype=torch.float16)
            except Exception as e:
                print(f"Failed to load or process image from {image_path}: {e}")
                continue

            inp = item['conversations'][0]['value'].replace('<image>', '')
            true_values = parse_nutritional_info(item['conversations'][1]['value'], log_file, item)
            if true_values is None:
                continue

            for key in true_values:
                totals[key] += true_values[key]
                counts[key] += 1

            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
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
                predicted_values = parse_nutritional_info(outputs, log_file, item)
                if predicted_values is None:
                    continue

                for key in true_values:
                    if key in predicted_values:
                        errors[key].append(abs(predicted_values[key] - true_values[key]))
                    else:
                        log_file.write(f"Key {key} missing in predicted values from image {image_path}\n")

    # 计算MAE以及其百分比
    for key in totals.keys():
        mean = totals[key] / counts[key] if counts[key] > 0 else 0
        mae = sum(errors[key]) / len(errors[key]) if errors[key] else 0
        mae_percent = (mae / mean * 100) if mean > 0 else 0
        print(f"{key}: Mean = {mean}, MAE = {mae}, MAE as a percent of mean = {mae_percent}%")

if __name__ == '__main__':
    main()