import requests
import json


def download_image(data, save_path):
    for item in data:
        url = item["url"]
        image_name = item["id"]
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(save_path + image_name + ".jpg", 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)


with open('/media/fast_data/huggingface/hub/datasets_own--FreedomIntelligence--ALLaVA-4V/snapshots/a86c85a0076c7b0e9c6c08a55ef314461d45db65/ALLaVA-Caption-LAION-4V.json', 'r', encoding='gbk') as j:
    data = json.load(j)

save_path = '/media/fast_data/huggingface/hub/datasets_own--FreedomIntelligence--ALLaVA-4V'
download_image(data, save_path)