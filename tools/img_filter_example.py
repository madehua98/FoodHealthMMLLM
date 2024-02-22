from utils.file_utils import *
from PIL import Image
import torchvision.transforms as transforms
from datasets.common_cls_dataset import SquarePad
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
inp_size = (384, 384)
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(inp_size, interpolation=3),  # BICUBIC interpolation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])


jit_model_path = '/media/fast_data/tool_models/has_food_fv_gpu.pt'
device = torch.device("cuda")

model = torch.jit.load(jit_model_path).to(device)
for real_input_path in sorted(make_dataset('/media/fast_data/food_image_test')):
    real_input = transform(Image.open(real_input_path).convert('RGB')).unsqueeze(0).to(device)
    re = model(real_input)
    re = torch.softmax(re, dim=-1)
    hasFood = re[0, 1].item() > 0.5
    print(real_input_path, re, hasFood)
