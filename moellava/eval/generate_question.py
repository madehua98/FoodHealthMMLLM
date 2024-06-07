import json
from tqdm import tqdm
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
def write_jsonl(file, object):
    with open(file, mode='a+', encoding='utf-8') as fw:
        object = json.dumps(object)
        fw.write(object + '\n')

dir = '/mnt/data_llm/json_file/'
filename = '172_ingredient_test_prompt10.json'
question_file = '172_ingredient_questions.jsonl'
answer_file = '172_ingredient_answers.jsonl'
question_2k = read_json(dir+filename)
question_str = 'What is the dish in the picture made of?'

for question in tqdm(question_2k):
    data = {}
    question_id = question['id']
    text = 'What dish is this?'
    image = question['image']
    data['question_id'] = question_id
    data['image'] = image
    #data['text'] = question['conversations'][1]['value']
    data['text'] = question_str
    data['category'] = 'default'

    write_jsonl(dir+question_file, data)