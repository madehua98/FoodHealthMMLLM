import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
# name = "haoranxu/ALMA-7B"
# name = "haoranxu/ALMA-7B-R"
# name = "haoranxu/ALMA-13B-R"

for name in ["haoranxu/ALMA-7B-R", "haoranxu/ALMA-13B-R"]:

    print(name)
    # Load base model and LoRA weights
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')

    # Add the source sentence into the prompt template
    prompts = ["Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:",
               "Translate this from English to Chinese:\nEnglish: The multi-modality large language model is designed for the big AGI industry.\nChinese:",
               "Translate this from English to Chinese:\nEnglish: Chocolate Peanut Butter Protein Bars.\nChinese:", # 巧克力花生酱蛋白棒
               ]

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()
        # Translation
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=200, do_sample=True, temperature=0.6, top_p=0.9)
            # generated_ids = model.generate(input_ids=input_ids, num_beams=1, max_new_tokens=200, do_sample=False, temperature=0.3, top_p=0.9)
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(outputs)

    del model, tokenizer

from transformers import pipeline
for name in ["Unbabel/TowerInstruct-7B-v0.1", "Unbabel/TowerInstruct-13B-v0.1"]:

    print(name)
    pipe = pipeline("text-generation", model=name, torch_dtype=torch.bfloat16, device_map="auto")
    # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    prompts = [
        "Translate the following text from Chinese into English.\nChinese: 我爱机器翻译。\nEnglish:",
        "Translate the following text from English into Chinese.\nEnglish: The multi-modality large language model is designed for the big AGI industry.\nChinese:",
        "Translate the following text from English into Chinese.\nEnglish: Chocolate Peanut Butter Protein Bars.\nChinese:",
    ]
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
        print(outputs[0]["generated_text"])
    del pipe


'''

conda activate trt_ascend && export PYTHONPATH=/home/xuzhenbo/LLaMA-Factory-main && export all_proxy=127.0.0.1:7890
"haoranxu/ALMA-13B-R"
CUDA_VISIBLE_DEVICES=0,1 http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 OMP_NUM_THREADS=8 python -u tools/translate.py
"haoranxu/ALMA-7B-R"
CUDA_VISIBLE_DEVICES=0 http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 OMP_NUM_THREADS=8 python -u tools/translate.py
'''