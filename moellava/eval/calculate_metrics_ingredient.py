
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.metrics.distance import edit_distance
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
def calculate_edit_distance(text1, text2):
    return edit_distance(text1, text2)


def sort_jsonl_by_question_id(input_file, output_file):
    # 读取 JSONL 文件中的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        json_lines = [json.loads(line) for line in f]
    
    # 按照 question_id 对 JSON 对象进行排序
    sorted_json_lines = sorted(json_lines, key=lambda x: int(x['question_id']))
    
    # 将排序后的 JSON 对象写回到新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_obj in sorted_json_lines:
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

def read_jsonl(file):
    data_list = []
    with open(file, mode='r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data = json.loads(line)
            data_list.append(data)
    return data_list

def find_most_similar_word(target_word, word_list):
    min_distance = float('inf')
    most_similar_word = None
    for word in word_list:
        distance = calculate_edit_distance(target_word, word)
        if distance < min_distance:
            min_distance = distance
            most_similar_word = word
    return most_similar_word, min_distance

def evaluate_multiclass(y_true, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算精确率、召回率和F1值
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # 返回结果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def get_all_ingredient(filename):
    ingredient_set = set()
    ingredient_dict = {}
    with open(filename, mode='r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr.readlines()):
            line = line.replace('\n', '')
            ingredient_set.add(line.lower())
    for idx, ingredient in enumerate(ingredient_set):
        ingredient_dict[ingredient] = idx
    return ingredient_dict

def get_ingredient_from_gold_answer(text):
    # Step 1: Split the string by comma
    parts = text.split(',')
    # Step 2: Extract all parts except the last one
    extracted_parts = [part.strip() for part in parts[:-1]]
    
    # Step 3: Process the last part separately
    last_part = parts[-1].strip()
    
    # Step 4: Split the last part by the first "and" and add to the extracted parts
    if "and" in last_part:
        first_and_split = last_part.split('and', 1)  # Split only on the first occurrence of "and"
        extracted_parts.append(first_and_split[0].strip())
        extracted_parts.append(first_and_split[1].strip())
    else:
        extracted_parts.append(last_part)
    ingredients = [extracted_part.lower() for extracted_part in extracted_parts]
    return ingredients


def get_ingredient_from_model_answer(text, ingredient_dict):
    # Step 1: Split the string by comma
    parts = text.split(',')
    # Step 2: Extract all parts except the last one
    extracted_parts = [part.strip() for part in parts[:-1]]
    
    # Step 3: Process the last part separately
    last_part = parts[-1].strip()
    
    # Step 4: Split the last part by the first "and" and add to the extracted parts
    if "and" in last_part:
        first_and_split = last_part.split('and', 1)  # Split only on the first occurrence of "and"
        extracted_parts.append(first_and_split[0].strip())
        extracted_parts.append(first_and_split[1].strip())
    else:
        extracted_parts.append(last_part)
    ingredients = [extracted_part.lower() for extracted_part in extracted_parts]
    ingredients_convert = []
    for ingredient in ingredients:
        if ingredient in ingredient_dict.keys():
            ingredients_convert.append(ingredient)
        else:
            most_similar_word, distance = find_most_similar_word(ingredient, ingredient_dict.keys())
            ingredients_convert.append(most_similar_word)
    return ingredients_convert
    
def calculate_iou(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)
    
    # 计算交集和并集
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 计算IoU
    iou = len(intersection) / len(union)
    return iou 

def calculate_average(lst):
    if len(lst) == 0:
        return 0  # 防止除以0错误
    return sum(lst) / len(lst)

def to_multilabel_vector(ingredients_list, ingredient_index):
    vector = np.zeros(len(ingredient_index), dtype=int)
    for ingredient in ingredients_list:
        if ingredient in ingredient_index:
            vector[ingredient_index[ingredient]] = 1
    return vector.tolist()
def get_metrics(answer_file_path, gold_answer_file_path):
    answer_file = read_jsonl(answer_file_path)
    gold_answer_file = read_jsonl(gold_answer_file_path)
    ingredient_dict = get_all_ingredient(ingredient_file)
    idx = 0
    gold_ingredients = []
    for data in tqdm(gold_answer_file, total=(len(gold_answer_file))):
        ingredients = get_ingredient_from_gold_answer(data['text'])
        gold_ingredients.append(ingredients)

    pridiction_ingredients = []
    for data in tqdm(answer_file, total=len(answer_file)):
        ingredients = get_ingredient_from_model_answer(data['text'], ingredient_dict)
        pridiction_ingredients.append(ingredients)

    y_true = np.array([to_multilabel_vector(ingredients, ingredient_dict) for ingredients in tqdm(gold_ingredients)])
    y_pred = np.array([to_multilabel_vector(ingredients, ingredient_dict) for ingredients in tqdm(pridiction_ingredients)])

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro F1:', micro_f1)

    # 计算宏平均F1值
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print('Macro F1:', macro_f1)

    # 计算样本平均F1值
    sample_f1 = f1_score(y_true, y_pred, average='samples')
    print('Sample F1:', sample_f1)

    # 计算每个类别的F1值
    f1_per_class = f1_score(y_true, y_pred, average=None)
    print('F1 per class:', f1_per_class)

    assert len(gold_ingredients) == len(pridiction_ingredients)
    iou_list = []
    for i in tqdm(range(len(gold_ingredients)), total=len(gold_ingredients)):
        iou = calculate_iou(gold_ingredients[i], pridiction_ingredients[i])
        iou_list.append(iou)

    iou_number = calculate_average(iou_list)
    print('iou为:', iou_number)
    return iou_number


# if __name__ == "__main()__":
gold_answer_file_path = "/mnt/data_llm/json_file/172_ingredient_answers.jsonl"
ingredient_file = '/media/fast_data/VireoFood172/SplitAndIngreLabel/IngredientList.txt'
answer_file_path = "/home/data_llm/madehua/FoodHealthMMLLM/eval/food172_ingredient/answers/MoE-LLaVA-Phi2-2.7B-4e-v172_ingredient_0426-checkpoint-1376.jsonl"
sorted_answer_file_path = "/home/data_llm/madehua/FoodHealthMMLLM/eval/food172_ingredient/answers/MoE-LLaVA-Phi2-2.7B-4e-v172_ingredient_0426-checkpoint-1376.jsonl"
sort_jsonl_by_question_id(answer_file_path, sorted_answer_file_path)
iou_number = get_metrics(sorted_answer_file_path, gold_answer_file_path)
print(iou_number)