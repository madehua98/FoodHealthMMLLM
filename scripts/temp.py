from utils.file_utils import *

# new = load_json("/media/fast_data/huggingface/hub/datasets_own--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/unify_la_tune_256k.json")
# ori = load_json("/media/fast_data/huggingface/hub/datasets_own--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json/la_tune_256k.json")

for json_path in make_dataset("/media/fast_data/huggingface/hub/datasets_own--LanguageBind--MoE-LLaVA/snapshots/78aa3747b7d65ec84486de39d3b644d7f863aa7b/train_json", suffix=".json"):
    info = load_json(json_path)
    print(json_path)
    # print(info[0])
    # print(info[1])
    # print("\n\n\n")
    if 'image' not in info[0].keys() or isinstance(info[0]['image'], str):
        continue
    for item in info:
        if isinstance(item['image'], list) and len(item['image']) > 1:
            print(item)