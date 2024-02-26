from utils.file_utils import *
device = torch.device("cuda")
model = torch.jit.load('/media/fast_data/tool_models/food_weight_fv_gpu.pt').to(device)
inp_size = (224, 224)

from PIL import Image
import torchvision.transforms as transforms
from datasets.common_cls_dataset import SquarePad
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(inp_size, interpolation=3),  # BICUBIC interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])
for real_input_path in ['/home/data_llm/new_fresh_devices2/*小花牛/*小花牛_002c27da-4db4-445b-aa54-2121124c4994_0.914-kg.jpg', '/home/data_llm/new_fresh_devices2/*小花牛/*小花牛_08a12700-f091-4064-9088-dd579b6c32de_0.436-kg.jpg', '/home/data_llm/new_fresh_devices2/*小花牛/*小花牛_0cc6c293-e8be-4995-8ef9-d893fd48a3cf_1.882-kg.jpg']:
    real_input = transform(Image.open(real_input_path).convert('RGB')).unsqueeze(0).to(device)
    re = model(real_input)
    print(real_input_path, re)

'''
python -u scripts/predict_food_weight.py
/home/data_llm/new_fresh_devices2/*小花牛/*小花牛_002c27da-4db4-445b-aa54-2121124c4994_0.914-kg.jpg tensor([[1.0760]], device='cuda:0', grad_fn=<AddmmBackward0>)
/home/data_llm/new_fresh_devices2/*小花牛/*小花牛_08a12700-f091-4064-9088-dd579b6c32de_0.436-kg.jpg tensor([[0.3999]], device='cuda:0', grad_fn=<AddmmBackward0>)
/home/data_llm/new_fresh_devices2/*小花牛/*小花牛_0cc6c293-e8be-4995-8ef9-d893fd48a3cf_1.882-kg.jpg tensor([[2.1736]], device='cuda:0', grad_fn=<AddmmBackward0>)
'''