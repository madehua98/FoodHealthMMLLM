import json
import os
import random
from tqdm import tqdm

def get_all_images_from_category(base_path, category):
    category_path = os.path.join(base_path, category)
    images = []
    for root, dirs, files in os.walk(category_path):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    return images

def split_categories(base_path):
    """随机选择80%的食物类别作为训练集，剩余20%作为测试集。"""
    all_categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    pcs_subcategories = [os.path.join('pcs_images', d) for d in os.listdir(os.path.join(base_path, 'pcs_images')) if os.path.isdir(os.path.join(base_path, 'pcs_images', d))]
    all_categories.extend(pcs_subcategories)
    random.shuffle(all_categories)
    split_point = int(len(all_categories) * 0.8)
    return all_categories[:split_point], all_categories[split_point:]

def limit_images_per_category(images, max_images=2000):
    """如果一个类别中的图片超过最大数量，随机挑选最大数量的图片。"""
    if len(images) > max_images:
        return random.sample(images, max_images)
    return images

def generate_conversations_and_json(images, output_file):
    dishes = []
    id_counter = 0

    for image_path in tqdm(images):
        parent_category = os.path.basename(os.path.dirname(image_path)).replace("_", " ").lower()
        weight_with_unit = os.path.basename(image_path).split('_')[-1].split('-')
        unit = "kg" if "pcs_images" not in image_path else ""

        if unit:
            try:
                weight = format(float(weight_with_unit[0]) * 1000, '.2f')
            except ValueError:
                print('Error,', image_path)
                continue
            prompt_options = [
                f"Estimate the mass (g) of the food in the image.",
                f"Estimate the mass (g) of the {parent_category} in the image."
                f"What are the estimated grams of mass in the food item depicted in this image?",
                f"What are the estimated grams of mass in the {parent_category} depicted in this image?",
                f"Can you analyze the food in this image and provide estimates for its mass (g)?",
                f"Can you analyze the {parent_category} in this image and provide estimates for its mass (g)?",
                f"Please provide the estimation of the mass (g) in the food shown in this image.",
                f"Please provide the estimation of the mass (g) in the {parent_category} shown in this image.",
                f"I need detailed information on the mass (g) of the food shown in this image. Can you provide that?",
                f"I need detailed information on the mass (g) of the {parent_category} shown in this image. Can you provide that?",
            ]
            response = f"{weight} g"
        else:
            try:
                weight = float(weight_with_unit[0])
            except ValueError:
                print('Error,', image_path)
                continue
            prompt_options = [
                "Estimate how many items are in the image.",
                f"Estimate how many {parent_category} are in the image."
            ]
            response = f"{weight}"

        prompt = random.choice(prompt_options)

        dishes.append({
            "id": str(id_counter),
            "image": image_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": response}
            ]
        })
        id_counter += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dishes, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    base_path = '/mnt/data_llm/new_fresh_devices2'
    output_train_file = '/media/fast_data/food_weight_predict_code/weight_dataset_train_0426.json'
    output_test_file = '/media/fast_data/food_weight_predict_code/weight_dataset_test_0426.json'
    train_categories, test_categories = split_categories(base_path)

    train_images = sum([limit_images_per_category(get_all_images_from_category(base_path, cat)) for cat in train_categories], [])
    test_images = sum([limit_images_per_category(get_all_images_from_category(base_path, cat)) for cat in test_categories], [])
    generate_conversations_and_json(train_images, output_train_file)
    generate_conversations_and_json(test_images, output_test_file)