import os
import json
import logging
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from faster_vit import faster_vit_1_224, faster_vit_2_224, faster_vit_3_224, faster_vit_4_224, faster_vit_5_224, faster_vit_6_224
from torch.optim.lr_scheduler import StepLR
from utils.file_utils import load_json, save_json, load_pickle, save_pickle
import random
from tqdm import tqdm
# 设置日志记录
logging.basicConfig(filename='training_classification1.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s')

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 读取训练数据
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

all_data = load_json('/media/fast_data/result.json')
all_data = list(all_data.items())
all_data_count = len(all_data)
train_data_count = int(len(all_data) * 0.8)
test_data_count = len(all_data) - train_data_count
train_data = random.sample(all_data, train_data_count)
test_data = [item for item in all_data if item not in train_data]

old_data_path = '/home/xuzhenbo/food_dataset/hasFood_samples.pkl'
old_data = load_pickle(old_data_path)
old_data_path1 = '/mnt/data_llm/hasFood_samples_v1.pkl'
pos_count = 0
neg_count = 0
not_exist = 0
old_data1 = {}
old_data1["train"] = {}
old_data1["test"] = {}
# print
count = 0
for key, value in tqdm(old_data['train'].items(), total=len(old_data['train'])):
    # if '/home/xuzhenbo/dishes_DaNeng/' not in key and '/home/xuzhenbo/food_dataset/food-101/' not in key and '/home/xuzhenbo/food_dataset/isia_500/ISIA_Food500/' not in key and '/home/xuzhenbo/food_dataset/isia_200/' not in key and '/home/xuzhenbo/dishes_DaNeng_val/' not in key:
    if not os.path.exists(key):
        if '/home/xuzhenbo/food_dataset/food-101/' in key:
            key = key.replace('/home/xuzhenbo/food_dataset/food-101/images', '/home/xuzhenbo/food_dataset/food-101/train')
    if count % 10 == 0:
        if os.path.exists(key):
            old_data1["train"][key] = value
            count += 1

count = 0
for key, value in tqdm(old_data['val'].items(), total=len(old_data['val'])):
    # if '/home/xuzhenbo/dishes_DaNeng/' not in key and '/home/xuzhenbo/food_dataset/food-101/' not in key and '/home/xuzhenbo/food_dataset/isia_500/ISIA_Food500/' not in key and '/home/xuzhenbo/food_dataset/isia_200/' not in key and '/home/xuzhenbo/dishes_DaNeng_val/' not in key:
    if count % 10 == 0:
        if os.path.exists(key):
            old_data1["test"][key] = value
            count += 1

img_dir = '/media/fast_data/image_hasfood_label/res_2_3_part1'

for (key, value) in train_data:
    path = os.path.join(img_dir, key)
    if value == "是":
        old_data1["train"][path] = 1
    if value == "否":
        old_data1["train"][path] = 0

for (key, value) in test_data:
    path = os.path.join(img_dir, key)
    if value == "是":
        old_data1["test"][path] = 1
    if value == "否":
        old_data1["test"][path] = 0

save_pickle(old_data_path1, old_data1)
# pos 169731  neg 160000
#print(old_data['train'])

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = list(self.data.keys())[idx]
        label = list(self.data.values())[idx]
        img_path = img_name  # 或者使用 os.path.join(self.img_dir, img_name) 如果 img_name 只是文件名
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, IOError) as e:
            # 尝试下一个图像
            return self.__getitem__((idx + 1) % len(self.data))

        if self.transform:
            image = self.transform(image)
        
        return image, label

# 自定义损失函数
class CustomClassificationLoss(nn.Module):
    def __init__(self):
        super(CustomClassificationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, outputs, targets):
        # 计算交叉熵损失
        ce_loss = self.ce_loss(outputs, targets)
        return ce_loss

# 加载和预处理数据
json_file = '/media/fast_data/result.json'
img_dir = '/media/fast_data/image_hasfood_label/res_2_3_part1'  # 图像文件夹路径
#data = load_data(json_file)
data = load_pickle(old_data_path1)
train_data = data["train"]
test_data = data["test"]
train_size = len(train_data)
val_size = len(test_data)

# 数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建数据集
dataset = ImageDataset(data, img_dir, transform=transform)
train_dataset = ImageDataset(train_data, img_dir, transform=transform)
val_dataset = ImageDataset(test_data, img_dir, transform=transform)


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=64)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=64)

# 定义模型
model = faster_vit_4_224(pretrained='/home/xuzhenbo/Downloads/fastervit_4_21k_224_w14.pth.tar')
model.head = nn.Linear(model.num_features, 2)

best_accuracy = 0.0

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

criterion = CustomClassificationLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-4)  
#scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # 每5个epoch将学习率降低到原来的10%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

num_epochs = 4
best_accuracy=0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    accurate_predictions = 0
    total_predictions = 0
    val_loss = 0
    for images, labels in tqdm(train_loader, total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)
        accurate_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    val_loss /= val_size
    accuracy = accurate_predictions / total_predictions
    logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')


    epoch_loss = running_loss / train_size
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

    # scheduler.step()

    # 验证模型
    model.eval()
    val_loss = 0.0
    accurate_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # 计算准确率
            predictions = torch.argmax(outputs, dim=1)
            accurate_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    val_loss /= val_size
    accuracy = accurate_predictions / total_predictions
    logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # 更新最佳准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), f'/media/fast_data/tool_models/hasfood_fastervit_4_21k_224.pth')


# # 使用最佳alpha重新训练模型
# criterion = CustomClassificationLoss(alpha=best_alpha)
# optimizer = optim.Adam(model.parameters(), lr=1e-4) 
# scheduler = StepLR(optimizer, step_size=2, gamma=0.1) 

# model.to(device)
# criterion.to(device)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * images.size(0)

#     epoch_loss = running_loss / train_size
#     logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
#     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

#     scheduler.step()

#     # 验证模型
#     model.eval()
#     val_loss = 0.0
#     accurate_predictions = 0
#     total_predictions = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * images.size(0)

#             # 将outputs转换为实际的百分比值
#             outputs_percentage = torch.argmax(outputs, dim=1).float() * 10
#             labels_percentage = labels.float() * 10
            
#             # 计算准确率
#             predictions = outputs_percentage.cpu().numpy()
#             true_values = labels_percentage.cpu().numpy()
#             accurate_predictions += np.sum(np.abs(predictions - true_values) < 10)  # 假设误差小于10%为准确预测
#             total_predictions += len(true_values)

#     val_loss /= val_size
#     accuracy = accurate_predictions / total_predictions
#     logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
#     print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

# # 保存模型
# torch.save(model.state_dict(), f'image_labeling_model_faster_vit_alpha_{best_alpha}_1.pth')
# logging.info(f'Model saved to image_labeling_model_faster_vit_alpha_{best_alpha}_1.pth')
# print(f'Model saved to image_labeling_model_faster_vit_alpha_{best_alpha}_1.pth')
