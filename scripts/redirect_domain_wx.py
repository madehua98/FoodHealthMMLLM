from flask import Flask, redirect, request, jsonify
from openai import OpenAI
import os, base64, requests
from PIL import Image
from io import BytesIO
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ['http_proxy'] = '127.0.0.1:7890'
my_api_key = "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"
url = "https://api.openai.com/v1/chat/completions"


app = Flask(__name__)

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
                        "text": "请根据给的食物图片，描述图片中的营养成分并给出饮食建议。"
                                "希望能够按照以下格式给出回答："
                                "膳食营养摄入情况：食物1（重量），食物2（重量），成分3（重量）……"
                                "饮食建议：……"
                                "例如："
                                "膳食营养摄入情况：咖喱（10g）、虾（50g）、米饭（200g）、烤馒头（100g）"
                                "饮食建议：缺少蔬菜，记得补充点~"
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
        "max_tokens": 1024
    }

    # 调用MultiModalConversation API
    response = requests.post(url, headers=headers, json=payload)

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8900)