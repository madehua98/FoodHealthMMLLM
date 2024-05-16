
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.metrics.distance import edit_distance
import json
from tqdm import tqdm

def calculate_edit_distance(text1, text2):
    return edit_distance(text1, text2)


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

def get_metrics(answer_file_path, gold_answer_file_path):
    answer_file = read_jsonl(answer_file_path)
    gold_answer_file = read_jsonl(gold_answer_file_path)
    answer_dict = {}
    idx = 0

    for data in gold_answer_file:
        if data['text'] not in answer_dict.keys():
            answer_dict[data['text']] = idx
            idx += 1
    y_true = []
    for i in range(len(gold_answer_file)):
        y = answer_dict[gold_answer_file[i]['text']]
        y_true.append(y)
    y_pred = []
    for i in tqdm(range(len(answer_file))):
        if answer_file[i]['text'] in answer_dict.keys():
            y = answer_dict[answer_file[i]['text']]
            y_pred.append(y)
        else:
            answer, _ = find_most_similar_word(answer_file[i]['text'], list(answer_dict.keys()))
            y = answer_dict[answer]
            y_pred.append(y)
    
    metrics = evaluate_multiclass(y_true, y_pred)
    accuracy = metrics['accuracy'] 
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']
    return accuracy, precision, recall, f1_score


# if __name__ == "__main()__":
gold_answer_file_path = "/mnt/data_llm/json_file/101_answers.jsonl"
answer_file_path = "/home/data_llm/madehua/FoodHealthMMLLM/eval/food101/answers/MoE-LLaVA-Phi2-2.7B-4e-v101_0426_prompt1-checkpoint-1773-1.jsonl"
accuracy, precision, recall, f1 = get_metrics(answer_file_path, gold_answer_file_path)
print(accuracy, precision, recall, f1)
# file_dir = '/home/data_llm/madehua/FoodHealthMMLLM/eval/food101/answers/'
# filename = 'MoE-LLaVA-Phi2-2.7B-4e-v101_0426_prompt1-checkpoint-1773.jsonl'
# file_path = file_dir + filename
# data_old = []
# with open(file_path, mode='r', encoding='utf-8') as fr:
#     for data in fr.readlines():
#         data = json.loads(data)
#         data_old.append(data)
# data_new = []
# id_store = []
# for data in data_old:
#     id = data['question_id']
#     if id not in id_store:
#         id_store.append(id)
#         data_new.append(data)
# filename1 = 'MoE-LLaVA-Phi2-2.7B-4e-v101_0426_prompt1-checkpoint-1773-1.jsonl'
# file_path1 = file_dir + filename1
# for d in data_new:
#     with open(file_path1, mode="a+", encoding='utf-8') as fw:
#         fw.write(json.dumps(d, ensure_ascii=False))
#         fw.write('\n')