import json
import os
import random
from  tqdm import tqdm
dict_template = {"id": "none", "image": "image_path", "conversations":[]}

food_recognition_prompts = [
    #"What is the name of this dish?",
    #"What is the name of the dish shown in the image",
    # "Can you tell me the dish's name?",
    "What dish is this?",
    # "Can you tell me the name of this dish?",
    # "What is the culinary name of this dish?",
    # "Can you provide the name of the dish?",
    # "What is the category of the dish presented in the image?",
    #"Can you identify the dish displayed in the photo?",
    # "Which dish is depicted in the picture?"
]
ingredient_recognition_prompts = [
    # "Can you identify the ingredients present in this image?",
    # "What ingredients are visible in this picture?",
    # "Can you identify the ingredients present in this image?",
    # "Which food ingredients can you discern in this photo?",
    # "Can you identify the ingredients from this picture?",
    # "What are the ingredients of the dish depicted in the image?",
    # "Can you list the components of the dish shown in the photo?",
    "What is the dish in the picture made of?",
]

def find_image_file(directory_path):
    jpg_files = []
    for file in os.listdir(directory_path):
        if file.endswith(".jpg"):
            jpg_files.append(file)
    return jpg_files

def write_json_file(data, json_file_path):
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def get_datas_from_split(split_file_path):
    with open(split_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    paths = []
    names = []
    for line in lines:
        line = line.replace("\n","")
        path = line.split("/")
        paths.append(path)
        name = path[0].split("_")
        name = " ".join(name)
        name = name.capitalize()
        names.append(name)
    return paths, names

def food2k_get_paths_from_split(
        split_fn,
) -> list[str] and list[int]:
    with open(split_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    paths = []
    label_ids = []
    for line in lines:
        line = line.replace("\n","")
        path = line
        paths.append(path)
        label_id = path.split("/")[1]
        label_ids.append(label_id)
    return paths, label_ids

def food172_get_paths_from_split(
        split_fn,
) -> list[str] and list[int]:
    with open(split_fn, "r", encoding="utf-8") as f:
        lines = f.readlines()
    paths = []
    label_ids = []
    for line in lines:
        line = line.replace("\n","")
        path = line
        paths.append(path)
        label_id = path.split("/")[1]
        label_ids.append(label_id)
    return paths, label_ids


def food2k_get_labels_from_file(label_file_path):
    with open(label_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    labels = {}
    for line in lines:
        parts = line.split("--")
        number = int(parts[0])
        dish_name = parts[1].replace("\n", "")
        labels.update({str(number):dish_name})
    return labels

def food101_get_labels_from_file(label_file_path):
    with open(label_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    labels_fn = {}
    labels = {}
    for index, line in enumerate(lines):
        line = line.replace("\n", "")
        names = line.split(" ")
        dish_name_fn = "_".join(names)
        dish_name = " ".join(names)
        labels_fn.update({str(index):str.lower(dish_name_fn)})
        labels.update({str(index):dish_name})
    return labels_fn, labels

def food172_get_labels_from_file(label_file_path):
    with open(label_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    labels = {}
    for index, line in enumerate(lines):
        line = line.replace("\n", "").replace(";","")
        line = line.capitalize()
        labels[index+1] = line
    return labels

def food172_ingredient_get_labels_from_file(label_file_path):
    with open(label_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    labels = {}
    for index, line in enumerate(lines):
        line = line.replace("\n", "").replace(";","")
        line = line.capitalize()
        labels[index] = line
    return labels

def food172_ingredient_labels_from_file(label_file_path):
    with open(label_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    image_labels = {}
    for index, line in enumerate(lines):
        line = line.replace("\n", "").replace(";","")
        line = line.split(".jpg ")
        image = str(line[0]) + ".jpg"
        image_label = line[1].split(" ")
        del image_label[0]
        indexes_of_1 = [index for index, value in enumerate(image_label) if value == "1"]
        image_labels[image] = indexes_of_1
    return image_labels

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

def food2k_trans(file_path, label_file_path, json_file_path):
    index = 0
    data = []
    labels = food2k_get_labels_from_file(label_file_path)
    for i in tqdm(range(2000)):
        jpg_files = find_image_file(file_path+f"/{i}")
        for jpg_file in jpg_files:
            image_path = file_path+f"/{i}/{jpg_file}"
            label = labels[str(i)]
            for prompt in food_recognition_prompts:
                gen_data(prompt, label, index, image_path, data)
                index += 1
    write_json_file(data, json_file_path)

def food2k_split_trans(file_path, label_file_path, json_file_path):
    index = 0
    data = []
    paths, label_ids = food2k_get_paths_from_split(file_path)
    labels = food2k_get_labels_from_file(label_file_path)
    for idx, path in tqdm(enumerate(paths)):
        image_path = food2k_file_path + path
        label = labels[str(label_ids[idx])]
        for prompt in food_recognition_prompts:
            gen_data(prompt, label, index, image_path, data)
            index += 1
    write_json_file(data, json_file_path)

def food101_trans(file_path, label_file_path, json_file_path):
    index = 0
    data = []
    labels_fn, labels = food101_get_labels_from_file(label_file_path)
    for i in range (101):
        file_name = labels_fn[str(i)]
        jpg_files = find_image_file(file_path+f"/{file_name}")
        for jpg_file in jpg_files:
            image_path = file_path+ f"/{file_name}/{jpg_file}"
            label = labels[str(i)]
            for prompt in food_recognition_prompts:
                gen_data(prompt, label, index, image_path, data)
                index += 1
    write_json_file(data, json_file_path)

def food101_split_trans(file_path,
                        label_file_path,
                        json_file_path,
                        split_path
                        ):
    data = []
    paths, names = get_datas_from_split(split_path)
    for index, path in enumerate(paths):
        dict_template = {}
        image_path = file_path+ f"/{path[0]}/{path[1]}.jpg"
        label = names[index]
        for prompt in food_recognition_prompts:
            gen_data(prompt, label, index, image_path, data)
            index += 1
    write_json_file(data, json_file_path)

def food172_trans(file_path, label_file_path, json_file_path):
    index = 0
    data = []
    labels = food172_get_labels_from_file(label_file_path)
    for i in range (1,173):
        jpg_files = find_image_file(file_path+f"/{i}")
        for jpg_file in jpg_files:
            image_path = file_path+f"/{i}/{jpg_file}"
            label = labels[i]
            for prompt in food_recognition_prompts:
                gen_data(prompt, label, index, image_path, data)
                index += 1
    write_json_file(data, json_file_path)


def food172_split_trans(file_path,
                        label_file_path,
                        json_file_path,
                        split_path
                        ):
    index = 0
    data = []
    labels = food172_get_labels_from_file(label_file_path)
    paths, label_ids = food172_get_paths_from_split(split_path)
    for index, path in enumerate(paths):
        image_path = file_path+ f"{path}"
        label_id = label_ids[index]
        label = labels[int(label_id)]
        for prompt in food_recognition_prompts:
            gen_data(prompt, label, index, image_path, data)
            index += 1
    write_json_file(data, json_file_path)


def format_string_complex(input_string):
    # First, split the string by the full-width comma
    parts = input_string.split("，")
    # Lowercase all parts except for the first one
    formatted_parts = [parts[0]] + [s.lower() for s in parts[1:]]
    # Join the parts with half-width commas except for the last one, which is joined with "and"
    formatted_string = ", ".join(formatted_parts[:-1]) + " and " + formatted_parts[-1]
    return formatted_string

def food172_ingredient_trans(image_fn, label_fn, ingredient_label_fn, json_fn):
    index = 0
    data = []
    labels = food172_ingredient_get_labels_from_file(label_fn)
    ingredient_labels = food172_ingredient_labels_from_file(ingredient_label_fn)
    for image_path, label_list in ingredient_labels.items():
        dict_template = {}
        image_path = image_fn+f"{image_path}"
        label_l = []
        for label_id in label_list:
            label_l.append(labels[label_id])
        label = ", ".join(label_l)
        if len(label_l) > 1:
            parts = label.split(", ")
            processed_parts = [parts[0]] + [part.lower() for part in parts[1:-1]] + ["and " + parts[-1].lower()]
            label = "，".join(processed_parts)
        for prompt in ingredient_recognition_prompts:
            gen_data(prompt, label, index, image_path, data)
            index += 1
    write_json_file(data, json_fn)


def food172_ingredient_split_trans(image_fn, label_fn, ingredient_label_fn, json_fn, split_fn):
    index = 0
    data = []
    labels = food172_ingredient_get_labels_from_file(label_fn)
    paths, label_ids = food172_get_paths_from_split(split_fn)
    ingredient_labels = food172_ingredient_labels_from_file(ingredient_label_fn)
    for index, path in enumerate(paths):
        dict_template = {}
        image_path = image_fn+ f"{path}"
        label_list = ingredient_labels[path]
        label_l = []
        for label_id in label_list:
            label_l.append(labels[label_id])
        label = ", ".join(label_l)
        if len(label_l) > 1:
            parts = label.split(", ")
            processed_parts = [parts[0]] + [part.lower() for part in parts[1:-1]] + ["and " + parts[-1].lower()]
            label = "，".join(processed_parts)
        for prompt in ingredient_recognition_prompts:
            gen_data(prompt, label, index, image_path, data)
            index += 1
    write_json_file(data, json_fn)



food2k_json_file_path = "/media/fast_data/json_file/2k_test_prompt1.json"
food2k_file_path = "/media/fast_data/Data/Food2k_complete"
food2k_split_file_path = "/media/fast_data/Data/Food2k_complete/test.txt"
food2k_label_file_path = "/media/fast_data/Data/Food2k_complete/food2k_label2name_en.txt"
#food2k_trans(food2k_file_path, food2k_label_file_path, food2k_json_file_path)
food2k_split_trans(food2k_split_file_path, food2k_label_file_path, food2k_json_file_path)

food101_json_file_path = "/media/fast_data/json_file/101_prompt1.json"
food101_train_json_fn = "/media/fast_data/json_file/101_val_prompt1.json"
food101_file_path = "/media/LLM_data/food_recognition_dataset/food-101/images"
food101_label_file_path = "/media/LLM_data/food_recognition_dataset/food-101/meta/labels.txt"
food101_train_file_path = "/media/LLM_data/food_recognition_dataset/food-101/meta/val.txt"
#food101_trans(food101_file_path, food101_label_file_path, food101_json_file_path)
# food101_split_trans(file_path=food101_file_path,
#                     label_file_path=food101_label_file_path,
#                     json_file_path=food101_train_json_fn,
#                     split_path=food101_train_file_path)

food172_json_fn = "/media/fast_data/json_file/172_prompt1.json"
food172_train_json_fn = "/media/fast_data/json_file/172_val_prompt1.json"
food172_fn = "/media/LLM_data/food_recognition_dataset/VireoFood172/ready_chinese_food"
food172_label_file_path = "/media/LLM_data/food_recognition_dataset/VireoFood172/SplitAndIngreLabel/FoodList.txt"
food172_train_fn = "/media/LLM_data/food_recognition_dataset/VireoFood172/SplitAndIngreLabel/VAL.txt"
#food172_trans(food172_fn, food172_label_file_path, food172_json_fn)
#food172_split_trans(food172_fn, food172_label_file_path, food172_train_json_fn, food172_train_fn)


food172_ingredient_fn = "/media/LLM_data/food_recognition_dataset/VireoFood172/SplitAndIngreLabel/IngredientList.txt"
food172_ingredient_label_fn = "/media/LLM_data/food_recognition_dataset/VireoFood172/SplitAndIngreLabel/IngreLabel.txt"
food172_image_fn = "/media/LLM_data/food_recognition_dataset/VireoFood172/ready_chinese_food"
food172_ingredient_json_fn = "/media/fast_data/json_file/172_ingredient_prompt1.json"
food172_ingredient_split_json_fn = "/media/fast_data/json_file/172_ingredient_val_prompt1.json"
food172_train_fn = "/media/LLM_data/food_recognition_dataset/VireoFood172/SplitAndIngreLabel/VAL.txt"


#food172_ingredient_trans(food172_image_fn, food172_ingredient_fn, food172_ingredient_label_fn, food172_ingredient_json_fn)


#food172_ingredient_split_trans(food172_image_fn, food172_ingredient_fn, food172_ingredient_label_fn, food172_ingredient_split_json_fn, food172_train_fn)


# def read_json_file(json_path):
#     with open(json_path, "r", encoding="utf-8") as fp:
#         data_list = []
#         data = json.load(fp)
#     return data
#
# food101_data_path ="/media/LLM_data/food_recognition_dataset/food-101/data.json"
# food101_data = read_json_file(food101_data_path)
# data_list = []
# for inx, data in enumerate(food101_data):
#     if inx % 1000 == 0:
#         data_list.append(data)

#write_json_file(data_list, "/media/LLM_data/food_recognition_dataset/food-101/food-101_data_sample.json")