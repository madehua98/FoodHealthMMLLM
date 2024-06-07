from tqdm import tqdm
import json

file_dir = '/media/fast_data/VireoFood172/SplitAndIngreLabel/'
filename = 'IngreLabel.txt'
ingredientlist_filename= 'IngredientList.txt'
train_filename = 'TR.txt'
test_filename = 'TE.txt'
val_filename = 'VAL.txt'

ingredient_recognition_prompts = [
    "Can you identify the ingredients present in this image?",
    "What ingredients are visible in this picture?",
    "Can you identify the ingredients present in this image?",
    "Which food ingredients can you discern in this photo?",
    "Can you identify the ingredients from this picture?",
    "What are the ingredients of the dish depicted in the image?",
    "Can you list the components of the dish shown in the photo?",
    "What is the dish in the picture made of?",
]

def write_json_file(data, json_file_path):
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def convert_to_format(input_str):
    parts = input_str.split()
    image_path = parts[0]
    values = parts[1:]
    
    positions = [ingredient_label[index] for index, value in enumerate(values) if value == '1']
    formatted_dict.update({image_path: positions})

def read_txt(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            datas.append(line)
    return datas

def conversation(prompt, label):
    conversation1 = {"from": "human","value": f"<image>\n{prompt}"}
    conversation2 = {"from": "gpt","value": f"{label}"}
    return conversation1, conversation2

def gen_data(prompt, label, index, image_path, data):
    conversation1, conversation2 = conversation(prompt=prompt,
                                                label=label)
    dict_template = {}
    dict_template["id"] = str(index)
    dict_template["image"] = image_path
    dict_template["conversations"] = [conversation1, conversation2]
    data.append(dict_template)

image_ingredients = read_txt(file_dir + filename)
train_image = read_txt(file_dir + train_filename)
test_image = read_txt(file_dir + test_filename)
val_image = read_txt(file_dir + val_filename)
train_image.extend(val_image)

ingredient_label = read_txt(file_dir + ingredientlist_filename)
ingredient_label_dict = {}
for idx, data in enumerate(ingredient_label):
    ingredient_label_dict[idx] = data
formatted_dict = {}
for data in image_ingredients:
    convert_to_format(data)


train_json_fn = "/media/fast_data/VireoFood172/172_ingredient_train_prompt10.json"
test_json_fn = "/media/fast_data/VireoFood172/172_ingredient_test_prompt10.json"

train_data = []
test_data = []
train_index = 0
test_index = 0
#print(train_image)
# for key, values in tqdm(formatted_dict.items()):
#     if len(values) == 1:
#         formatted_words = [values[0].capitalize().replace('\n', '')]
#         ingredient_str = formatted_words[0]
#     else:
#         formatted_words = [values[0].capitalize().replace('\n', '')] + [word.lower().replace('\n', '') for word in values[1:]]
#         ingredient_str = ', '.join(formatted_words[:-1]) + ' and ' + formatted_words[-1]
#     image_path = '/media/fast_data/VireoFood172/ready_chinese_food' + key
#     #print(key)
#     #print(train_image)
#     if key + '\n' in train_image:
#         prompt = ingredient_recognition_prompts[train_index%8]
#         gen_data(prompt, ingredient_str, train_index, image_path, train_data)
#         train_index += 1
#     elif key + '\n' in test_image:
#         prompt = ingredient_recognition_prompts[test_index%8]
#         gen_data(prompt, ingredient_str, test_index, image_path, test_data)
#         test_index += 1
    
#     write_json_file(train_data, train_json_fn)
#     write_json_file(test_data, test_json_fn)


for key in tqdm(test_image):
    key = key.replace('\n', '')
    values = formatted_dict[key]
    if len(values) == 1:
        formatted_words = [values[0].capitalize().replace('\n', '')]
        ingredient_str = formatted_words[0]
    else:
        formatted_words = [values[0].capitalize().replace('\n', '')] + [word.lower().replace('\n', '') for word in values[1:]]
        ingredient_str = ', '.join(formatted_words[:-1]) + ' and ' + formatted_words[-1]
    image_path = '/media/fast_data/VireoFood172/ready_chinese_food' + key
    prompt = ingredient_recognition_prompts[train_index%8]
    gen_data(prompt, ingredient_str, train_index, image_path, train_data)
    train_index += 1
write_json_file(train_data, test_json_fn)
    
    
# image_id = '/100/3_29.jpg'
# print(f"{image_id}的成分为{formatted_dict[image_id]}")
# image_id = '/100/10_29.jpg'
# print(f"{image_id}的成分为{formatted_dict[image_id]}")
# image_id = '/172/9.jpg'
# print(f"{image_id}的成分为{formatted_dict[image_id]}")
# image_id = '/172/9_51.jpg'
# print(f"{image_id}的成分为{formatted_dict[image_id]}")
# image_id = '/102/6_10.jpg'
# print(f"{image_id}的成分为{formatted_dict[image_id]}")