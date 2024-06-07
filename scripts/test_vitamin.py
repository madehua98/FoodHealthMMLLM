import torch
import open_clip
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    'jienengchen/ViTamin-XL-384px',
    trust_remote_code=True).to(device).eval()

image = Image.open('./docs/vitamin.png').convert('RGB')
image_processor = CLIPImageProcessor.from_pretrained('jienengchen/ViTamin-XL-384px')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
text = tokenizer(["a photo of vitamin", "a dog", "a cat"]).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features, text_features, logit_scale = model(pixel_values, text)
    text_probs = (100.0 * image_features @ text_features.to(torch.float).T).softmax(dim=-1)

print("Label probs:", text_probs)