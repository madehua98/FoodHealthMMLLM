import json

def write_jsonl(file, object):
    with open(file, mode='a+', encoding='utf-8') as fw:
        fw.write(json.dumps(object))
        fw.write('\n')

def read_json(file):
    with open(file, mode='r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def write_json(file, object):
    with open(file, mode='w', encoding='utf-8') as fw:
        json.dump(fp=fw, obj=object, ensure_ascii=False)


    
# print(data)

file_dir = '/mnt/data_llm/json_file/'
filename = '101_test_prompt10.json'
filename_questions = "101_questions_prompt10.jsonl"
file_path_questions = file_dir + filename_questions
file_path = file_dir + filename
with open(file_path, mode='r', encoding='utf-8') as fr:
    data = json.load(fr)

for d in data:
    line = dict(
        question_id=d["id"],
        image=d["image"],
        text=d["conversations"][0]["value"].replace('<image>\n', ''),
        category="default"
        )
    write_jsonl(file=file_path_questions, object=line)