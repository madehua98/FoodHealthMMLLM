import os
from openai import OpenAI
import base64
import requests
from PIL import Image
from io import BytesIO

# url = "http://127.0.0.1:8900/haohaochifan"
# image_path = "/home/xuzhenbo/MoE-LLaVA/docs/meal.jpg"

url = "http://118.25.98.83:8900/haohaochifan"
image_path = "/Users/jacksonxu/Downloads/FoodHealthMMLLM/docs/meal.jpg"


def encode_image(image_path):# 确保图片的尺寸不太大，长边不超过480
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode('utf-8')
    img = Image.open(image_path)
    w, h = img.size
    ratio = 480 / max(w, h)
    img = img.resize((int(w * ratio), int(h * ratio)))
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    return base64.b64encode(byte_data).decode('utf-8')

base64_image = encode_image(image_path)

headers = {"Content-Type": "application/json"}

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
    "max_tokens": 1024
}

print(payload)
try:
    response = requests.post(url, headers=headers, json=payload)
    print(response.json())
except Exception as ex:
    print("Exception:", ex)

