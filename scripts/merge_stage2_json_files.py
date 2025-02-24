# 打乱，调整比例，融合成一个单独的QA file
''' # json_folder=/mnt/data_llm/json_file
nutrition5k修改后训练的json路径：/mnt/data_llm/json_file/nutrition5k_train_modified1.json
nutrition5k修改后测试的json路径：/mnt/data_llm/json_file/nutrition5k_test_modified1.json
（修改了prompt，10个prompt随机选择）
${json_folder}/101_train_prompt10.json
${json_folder}/2k_train_prompt10.json
${json_folder}/172_train_prompt10.json
${json_folder}/172_ingredient_train_prompt10.json
train_recipe_QA.json
mix_food.json
'''
#
from utils.file_utils import *
json_folder='/mnt/data_llm/json_file'
datasets = ['mix_food.json', '101_train_prompt10.json', '2k_train_prompt10.json', '172_train_prompt10.json', '172_ingredient_train_prompt10.json', 'train_recipe_QA_10prompt.json', 'train_ingredient_QA_10prompt.json']
max_samples = 200000
min_len, max_len = 10, 1000

totalQA = []
save_loc = os.path.join(json_folder, 'stage2_240608.json')
save_loc1 = os.path.join(json_folder, 'stage2_240426.json')
for json_file in datasets:
    infos = load_json(os.path.join(json_folder, json_file))
    print(json_file, len(infos))

    # refine infos
    if 'mix_food' in json_file:
        lengths = [len(el["conversations"][0]["value"])+len(el["conversations"][1]["value"]) for el in infos]
        seq_index = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=False)
        seq_lengths = [lengths[el] for el in seq_index]
        seq_infos = [infos[el] for el in seq_index]
        infos = []
        for seq_len, info in tqdm(zip(seq_lengths, seq_infos), total=len(seq_lengths)):
            if min_len < seq_len < max_len:
                info['image'] = info["image"].replace('/media/fast_data/allava_data', '/mnt/data_llm/Mini-Gemini/Mini_Gemini_data/data/MGM-Pretrain/ALLaVA-4V/allava_data')
                assert os.path.isfile(info['image'])
                infos.append(info)
        print(len(seq_infos), len(infos))

    if '172_train_prompt10' in json_file or '101_train_prompt10.json' in json_file :
        res = infos + infos
    elif '2k' in json_file or 'ingredient' in json_file:
        res = infos
    elif len(infos) > max_samples:
        random.shuffle(infos)
        res = infos[:max_samples]
    else:
        res = infos
    totalQA += res
random.shuffle(totalQA)
print(len(totalQA)) # 2285223
save_json(save_loc, totalQA)
