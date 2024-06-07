import csv
import json
import os
from utils.file_utils import get_csv_content
from tqdm import tqdm

csvs = ['/Users/jacksonxu/Downloads/dish_metadata_cafe1.csv', '/Users/jacksonxu/Downloads/dish_metadata_cafe2.csv']
ingredients_file = '/Users/jacksonxu/Downloads/ingredients_metadata.csv'
error_relative_bound, error_absolute_bound = 0.18, 20

ingredients_info = get_csv_content(ingredients_file)
ingid2nameCalFatCarbProtein = {}
for content in ingredients_info[1:]:
    if len(content) < 6:
        print(content)
        continue
    name, ingid, cal, fat, carb, protein = content
    ingid2nameCalFatCarbProtein[int(ingid)] = [name, float(cal), float(fat), float(carb), float(protein)]

invalid_count = 0
for filepath in csvs:
    print(filepath)
    with open(filepath.replace('.csv', '_modified.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        with open(filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(tqdm(reader)):
                # if row[0] == 'dish_1551567573':
                #     b=1
                valid_line = True
                dish_id, total_calories, total_mass, total_fat, total_carb, total_protein = row[:6]
                total_calories, total_mass, total_fat, total_carb, total_protein = float(total_calories), float(total_mass), float(total_fat), float(total_carb), float(total_protein)
                ingredients = row[6:]
                assert len(ingredients) % 7 == 0
                count_calories, count_mass, count_fat, count_carb, count_protein = 0.0, 0.0, 0.0, 0.0, 0.0
                for ingCount in range(0, len(ingredients), 7):
                    ingr_id, ingr_name, ingr_grams, ingr_calories, ingr_fat, ingr_carb, ingr_protein = ingredients[ingCount:ingCount+7]
                    ingr_grams, ingr_calories = float(ingr_grams), float(ingr_calories)

                    ingr_per_name, ingr_per_calories, ingr_per_fat, ingr_per_carb, ingr_per_protein = ingid2nameCalFatCarbProtein[int(ingr_id[5:])]
                    # if ingr_per_calories > 0: # just delete error data
                    #     estimated_mass = ingr_calories/ingr_per_calories
                    #     if ingr_grams/estimated_mass > 10:
                    #         ingr_grams = estimated_mass
                    #         print('Fixing ingr_grams from {} to {}'.format(ingr_grams, estimated_mass))

                    count_calories += float(ingr_calories)
                    count_mass += float(ingr_grams)
                    count_fat += float(ingr_fat)
                    count_carb += float(ingr_carb)
                    count_protein += float(ingr_protein)
                if count_calories>0 and (abs(count_calories - total_calories) / count_calories > error_relative_bound or abs(count_calories - total_calories)>error_absolute_bound): # big error
                    print('total_calories Changing from {} to {}'.format(total_calories, count_calories))
                    row[1] = count_calories
                    print(row)
                    if 'dish_metadata_cafe1' in filepath:
                        valid_line = False
                if count_mass>0 and (abs(count_mass - total_mass) / count_mass > error_relative_bound or abs(count_mass - total_mass) > error_absolute_bound): # big error
                    print('total_mass Changing from {} to {}'.format(total_mass, count_mass))
                    row[2] = count_mass
                    print(row)
                    if 'dish_metadata_cafe1' in filepath:
                        valid_line = False
                if count_fat>0 and (abs(count_fat - total_fat) / count_fat > error_relative_bound or abs(count_fat - total_fat) > error_absolute_bound): # big error
                    print('total_fat Changing from {} to {}'.format(total_fat, count_fat))
                    row[3] = count_fat
                    print(row)
                    if 'dish_metadata_cafe1' in filepath:
                        valid_line = False
                if count_carb > 0 and (abs(count_carb - total_carb) / count_carb > error_relative_bound or abs(count_carb - total_carb) > error_absolute_bound): # big error
                    print('total_carb Changing from {} to {}'.format(total_carb, count_carb))
                    row[4] = count_carb
                    print(row)
                    if 'dish_metadata_cafe1' in filepath:
                        valid_line = False
                if count_protein>0 and (abs(count_protein - total_protein) / count_protein > error_relative_bound or abs(count_protein - total_protein) > error_absolute_bound): # big error
                    print('total_protein Changing from {} to {}'.format(total_protein, count_protein))
                    row[5] = count_protein
                    print(row)
                    if 'dish_metadata_cafe1' in filepath:
                        valid_line = False
                if valid_line and count_mass > 1500:
                    print('too heavy', count_mass)
                    print(row)
                    valid_line = False
                if valid_line:
                    writer.writerow(row)
                else:
                    invalid_count += 1

print(invalid_count)
