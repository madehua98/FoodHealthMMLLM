import requests
import json

instructs = [
    "Is the street crowded with people? Yes，the street is filled with a considerable number of people, which",
    "What can be observed above the car that's drivingdown the street? There is a colorful sign above the car as",
    "What is the overall atmosphere of the areadepicted in the image? The overall atmosphere in the image",
    "What type of food is in the box in the image? There is a loaded pizza pie with toppings inside the box in..",
    "Where is the pizza box placed? The pizza box is placed on the grass.",
    "Is the pizza box open or closed? The pizza box is open, and the whole pizza with toppings is visible.",
    "What kind of environment or setting is the image taken in? Can you describe it? The image wastaken in an..",
    "What color is the bathroom in the image?\n The bathroom in the image is all white.",
    "Does the bathroom have a bathtub or a shower? The bathroom has a bathtub.",
]


# url = 'http://118.25.98.83:22225/v1/chat/completions'
url = 'http://127.0.0.1:8000/v1/chat/completions'

for instruct in instructs:
    instruct = "Please judge the following paragraph contains food or not. '" + instruct[:800] + "' If related to food, please answer yes, otherwise no。"
    # instruct = "请判断以下句子是否与食物或相关。句子内容是：" + instruct[:1000] + "。请回答是或者不是。"

    # instruct = "你好吗？"
    myobj = {
        "model": "Qwen/Qwen1.5-7B-Chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruct},
        ],
        # "do_sample": True,
        # "repetition_penalty": 2.0,
        # "frequency_penalty": 0.3,
        # "top_p": 0,
        # "n": 1,
        # "max_tokens": 2048,
        "stream": False
    }
    headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

    try:
        response = requests.post(url, data=json.dumps(myobj), headers=headers)
        print(response)
        print(myobj, '\n', response.json()["choices"][0]["message"]["content"].strip())
        # break
    except Exception as e:
        b=1