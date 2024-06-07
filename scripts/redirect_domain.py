# from flask import Flask, redirect, request, jsonify
# import os, base64, requests
# os.environ['https_proxy'] = '127.0.0.1:7890'
# os.environ['http_proxy'] = '127.0.0.1:7890'
# my_api_key = "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"
# url = "https://api.openai.com/v1/chat/completions"
#
# headers = {"Content-Type": "application/json", "Authorization": "Bearer " + my_api_key}
# app = Flask(__name__)
#
#
# def request_openai(payload):
#     # image_path = "/home/ubuntu/meal.jpg"
#     # base64_image = encode_image(image_path)
#     # payload = {
#     #     "model": "gpt-4-vision-preview",
#     #     "messages": [{
#     #         "role": "user",
#     #         "content": [
#     #         {
#     #             "type": "text",
#     #             "text": "What’s in this image?"
#     #         },
#     #         {
#     #             "type": "image_url",
#     #             "image_url": {
#     #             "url": f"data:image/jpeg;base64,{base64_image}"
#     #             }
#     #         }
#     #         ]
#     #     }],
#     #     "max_tokens": 512
#     # }
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         print(response.json())
#         res = response.json()['choices'][0]['message']
#     except Exception as ex:
#         print("Exception:", ex)
#         res = {'role': 'Exception', 'content': 'Exception'}
#     return jsonify(data=res)
#
# @app.route('/haohaochifan', methods=['POST'])
# def registrazione():
#     request_body = request.get_json()
#     return request_openai(request_body)
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8900)


import os
from openai import OpenAI
from flask import Flask, request, jsonify
import base64
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# url = "http://127.0.0.1:8900/haohaochifan"
# image_path = "/home/xuzhenbo/MoE-LLaVA/docs/meal.jpg"

url = "http://118.25.98.83:8900/haohaochifan"

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

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Save the image to a local file
        image_path = 'example.jpg'
        image_file.save(image_path)
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
            "max_tokens": 512
        }

        response = requests.post(url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Return only JSON serializable data
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to process image'})

    except Exception as ex:
        # Handle exceptions gracefully and return a valid response
        return jsonify({'error': str(ex)})

@app.route('/process_text', methods=['POST'])
def process_text():
    # 获取来自微信小程序的文本数据
    text = request.json['text']

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
            ]
        }],
        "max_tokens": 512
    }

    # 调用MultiModalConversation API
    response = requests.post(url, headers=headers, json=payload)

    return jsonify(response.json())


if __name__ == '__main__':
    app.run()
