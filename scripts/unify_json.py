import json
import os
from collections import Counter

mini_img_prefix = '/media/LLM_data/minigpt4v2_dataset/coco'

def multitask_conversation_trans():
    with open('/media/LLM_data/minigpt4v2_dataset/multitask_conversation/multitask_conversation.json', 'r') as f:
        original_data = json.load(f)
    transformed_data = []
    for item in original_data:
        transformed_item = {}
        first = {}
        second = {}
        transformed_item["id"] = int(item["id"])
        transformed_item["image"] = os.path.join(mini_img_prefix, item["image"])
        first["from"] = "human"
        second['from'] = "gpt"
        first['value'] = []
        second['value'] = []
        for i, item in enumerate(item["conversations"]):
            if i % 2 == 0:
                first['value'].append(item["value"])
            else:
                second['value'].append(item["value"])
        transformed_item["conversations"] = []
        transformed_item["conversations"].append(first)
        transformed_item["conversations"].append(second)
        transformed_data.append(transformed_item)

    with open('/media/fast_data/unified_json/unify_multitask_conversation.json',
              'w', encoding='utf-8') as jf:
        json.dump(transformed_data, jf, ensure_ascii=False, indent=2)



def okvqa_trans():
    with open('/media/LLM_data/minigpt4v2_dataset/okvqa/okvqa_train.json', 'r') as f:
        original_data = json.load(f)
    transformed_data = []
    for item in original_data:
        transformed_item = {}
        first = {}
        second = {}
        transformed_item["id"] = int(item["question_id"])
        transformed_item["image"] = os.path.join(mini_img_prefix, item["image"])
        first["from"] = "human"
        first['value'] = "<image>\n" + item["question"]
        second['from'] = "gpt"
        p = item["answer"]
        cp = Counter(p)
        ans, _ = cp.most_common(1)[0]
        second['value'] = ans
        transformed_item["conversations"] = []
        transformed_item["conversations"].append(first)
        transformed_item["conversations"].append(second)
        transformed_data.append(transformed_item)

    with open('/media/fast_data/unified_json/unify_okvqa_train.json',
              'w', encoding='utf-8') as jf:
        json.dump(transformed_data, jf, ensure_ascii=False, indent=2)


def aokvqa_trans():
    with open('/media/LLM_data/minigpt4v2_dataset/aokvqa/aokvqa_v1p0_train.json', 'r') as f:
        original_data = json.load(f)
    transformed_data = []
    for item in original_data:
        transformed_item = {}
        first = {}
        second = {}
        transformed_item["id"] = int(item["image_id"])
        transformed_item["image"] = os.path.join(mini_img_prefix, item["image"])
        first["from"] = "human"
        first['value'] = "<image>\n" + item["question"]
        second['from'] = "gpt"
        x = item["correct_choice_idx"]
        second['value'] = item["choices"][x]
        transformed_item["conversations"] = []
        transformed_item["conversations"].append(first)
        transformed_item["conversations"].append(second)
        transformed_data.append(transformed_item)

    with open('/media/fast_data/unified_json/unify_aokvqa_v1p0_train.json',
              'w', encoding='utf-8') as jf:
        json.dump(transformed_data, jf, ensure_ascii=False, indent=2)


okvqa_trans()
aokvqa_trans()
multitask_conversation_trans()


