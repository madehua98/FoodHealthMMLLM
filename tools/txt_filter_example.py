'''
slow
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import time
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-14B-Chat-AWQ",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat-AWQ")

def get_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Single sentence
prompt = "Give me a short introduction to large language model."
start = time.time()
for i in range(100):
    print(i)
    response = get_response(prompt)
    # print(response)
print(time.time() - start)
'''
export https_proxy=127.0.0.1:7890 && export http_proxy=127.0.0.1:7890
CUDA_VISIBLE_DEVICES=7 python -u tools/txt_filter_example.py
'''