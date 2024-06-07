import csv
import json
import os
import random

def read_ids_from_file(filepath):
    with open(filepath, 'r') as file:
        return set(file.read().splitlines())

def read_nutritional_info(dish_metadata_paths, train_ids, test_ids):
    nutritional_info_train = {}
    nutritional_info_test = {}
    for filepath in dish_metadata_paths:
        with open(filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                dish_id = row[0]
                nutrition_summary = {
                    "total_calories": format(float(row[1]), '.2f'),
                    "total_mass": format(float(row[2]), '.2f'),
                    "total_fat": format(float(row[3]), '.2f'),
                    "total_carbs": format(float(row[4]), '.2f'),
                    "total_protein": format(float(row[5]), '.2f')
                }
                if dish_id in test_ids:
                    nutritional_info_test[dish_id] = nutrition_summary
                elif dish_id in train_ids:
                    nutritional_info_train[dish_id] = nutrition_summary
    return nutritional_info_train, nutritional_info_test

def generate_json_data(imagery_base_path, nutritional_info):
    dishes = []
    dish_id_counter = 0
    prompts = [
        "Can you analyze the food in this image and provide estimates for its calories (kcal), mass (g), fat (g), carbs (g), and protein (g)?",
        "What is the nutritional content of the meal shown in this image, including calories (kcal), mass (g), fat (g), carbs (g), and protein (g)?",
        "Please calculate the calorie count (kcal) and the amounts of mass (g), fat (g), carbs (g), and protein (g) for the food depicted in this image.",
        "Could you estimate the nutritional values, including calories (kcal), mass (g), fat (g), carbs (g), and protein (g), of the dish in this image?",
        "I need detailed information on the calorie (kcal), mass (g), fat (g), carbs (g), and protein (g) content of the food shown in this image. Can you provide that?",
        "Assess and list the calories (kcal), mass (g), fat (g), carbs (g), and protein (g) for the meal captured in this image, please.",
        "How many calories (kcal) are in the food in this image? Also, provide the mass (g), fat (g), carbs (g), and protein (g) values.",
        "What are the estimated calories (kcal) and grams of mass, fat, carbs, and protein in the food item depicted in this image?",
        "Analyze the nutritional profile of the meal in this image, specifying the calories (kcal), mass (g), fat (g), carbs (g), and protein (g).",
        "Please provide a breakdown of the calorie (kcal) content and the mass (g), fat (g), carbs (g), and protein (g) in the food shown in this image."
    ]

    for dish_id, nutrition in nutritional_info.items():
        frames_folder_path = os.path.join(imagery_base_path, dish_id, 'frames_sampled5')
        if not os.path.exists(frames_folder_path):
            continue

        for image_file in os.listdir(frames_folder_path):
            image_path = os.path.join(frames_folder_path, image_file)
            nutritional_value = f"Calories: {nutrition['total_calories']}kcal, Mass: {nutrition['total_mass']}g, Fat: {nutrition['total_fat']}g, Carbs: {nutrition['total_carbs']}g, Protein: {nutrition['total_protein']}g"
            prompt = random.choice(prompts)

            conversations = [
                {
                    "from": "human",
                    "value": f"<image>\n{prompt}"
                },
                {
                    "from": "gpt",
                    "value": nutritional_value
                }
            ]

            dishes.append({
                "id": str(dish_id_counter),
                "image": image_path,
                "conversations": conversations
            })
            dish_id_counter += 1

    return json.dumps(dishes, indent=4)

if __name__ == "__main__":
    dish_metadata_paths = [
        '/media/fast_data/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1_modified.csv',
        '/media/fast_data/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2_modified.csv'
    ]
    train_ids_path = '/media/fast_data/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt'
    test_ids_path = '/media/fast_data/nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt'
    train_ids = read_ids_from_file(train_ids_path)
    test_ids = read_ids_from_file(test_ids_path)

    imagery_base_path = '/media/fast_data/nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles'

    nutritional_info_train, nutritional_info_test = read_nutritional_info(dish_metadata_paths, train_ids, test_ids)
    json_data_train = generate_json_data(imagery_base_path, nutritional_info_train)
    json_data_test = generate_json_data(imagery_base_path, nutritional_info_test)
    output_file_path_train = '/media/fast_data/food_weight_predict_code/nutrition5k_train_modified1.json'
    output_file_path_test = '/media/fast_data/food_weight_predict_code/nutrition5k_test_modified1.json'
    with open(output_file_path_train, 'w') as json_file1:
        json_file1.write(json_data_train)
        print(f"Data saved to {output_file_path_train}")
    with open(output_file_path_test, 'w') as json_file2:
        json_file2.write(json_data_test)
        print(f"Data saved to {output_file_path_test}")