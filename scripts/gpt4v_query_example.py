import os
from openai import OpenAI
import base64
import requests

# my_api_key = os.environ["OPENAI_API_KEY"]
my_api_key = "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"
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ['http_proxy'] = '127.0.0.1:7890'
# url = "https://api.openai.com/v1/chat/completions"
url = "http://127.0.0.1:22218/haohaochifan"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "/home/xuzhenbo/MoE-LLaVA/docs/meal.jpg"
base64_image = encode_image(image_path)

headers = {"Content-Type": "application/json", "Authorization": "Bearer " + my_api_key}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [{
        "role": "user",
        "content": [
        {
            "type": "text",
            "text": "What’s in this image?"
        },
        {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        ]
    }],
    "max_tokens": 512
}

try:
    response = requests.post(url, headers=headers, json=payload)
    print(response.json())
except Exception as ex:
    print("Exception:", ex)

