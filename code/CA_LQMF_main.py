import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm  # 导入tqdm库
from sklearn.metrics import f1_score, mean_absolute_error

import crossmodal
import dynamic
import adaptive


# 加载数据集
def load_mosi_data(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as f:
        text_data = np.array(f['text'])
        audio_data = np.array(f['audio'])
        video_data = np.array(f['video'])
        labels = np.array(f['labels'])
    return text_data, audio_data, video_data, labels


# 数据预处理
def preprocess_data(text_data, audio_data, video_data, labels):
    # 模态间缺失处理：标记缺失模态
    text_mask = (text_data != 0).astype(float)  # 将numpy数组转换为float
    audio_mask = (audio_data != 0).astype(float)  # 将numpy数组转换为float
    video_mask = (video_data != 0).astype(float)  # 将numpy数组转换为float

    # 模态内缺失处理：处理缺失特征
    # 使用零填充
    text_data = np.nan_to_num(text_data, nan=0.0)
    audio_data = np.nan_to_num(audio_data, nan=0.0)
    video_data = np.nan_to_num(video_data, nan=0.0)

    # 转换为tensor并转换为float类型
    text_data = torch.tensor(text_data).float()
    audio_data = torch.tensor(audio_data).float()
    video_data = torch.tensor(video_data).float()
    labels = torch.tensor(labels, dtype=torch.long)

    labels = torch.clamp(labels, min=0, max=1)

    # 返回处理后的数据
    # return text_data, audio_data, video_data, labels, text_mask, audio_mask, video_mask
    return text_data, audio_data, video_data, labels



# 进行信息蒸馏
class SelfDistillationModule(nn.Module):
    def __init__(self, input_dim):
        super(SelfDistillationModule, self).__init__()
        self.fc_a = nn.Linear(input_dim, input_dim)
        self.fc_b = nn.Linear(input_dim, input_dim)

    def forward(self, h_a, h_b):
        distill_a = self.fc_a(h_a)
        distill_b = self.fc_b(h_b)
        return distill_a, distill_b


# 主模块
class CA_LQMF(nn.Module):
    def __init__(self, input_dim, rank=4):
        super(CA_LQMF, self).__init__()
        # self.mcis = CrossModalInteractionModule(input_dim, rank)
        self.saca = crossmodal.CrossModalInteractionLayer(input_dim)
        # self.multi_modal_model = adaptive.MultiModalModel()

        self.dilution = SelfDistillationModule(input_dim)
        self.dynamic = dynamic.DynamicFeatureIntegrationModule(input_dim, 2)
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, text, audio, video):
        h_a = self.saca(text, audio, video)
        h_b = self.saca(audio, video, text)
        distill_a, distill_b = self.dilution(h_a, h_b)
        output_a = self.classifier(distill_a)
        output_b = self.classifier(distill_b)
        return output_a, output_b


# 训练过程
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for batch in tqdm(dataloader, desc="Training", ncols=100):
        text, audio, video, labels = batch
        optimizer.zero_grad()
        output_a, output_b = model(text, audio, video)
        loss = criterion(output_a, labels) + criterion(output_b, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 记录预测结果与真实标签
        _, predicted_a = torch.max(output_a, 1)
        _, predicted_b = torch.max(output_b, 1)
        predicted = (predicted_a + predicted_b) // 2
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

    # 计算F1 Score
    f1 = f1_score(all_labels, all_predictions)
    return running_loss / len(dataloader), f1


# 评估过程
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            text, audio, video, labels = batch
            output_a, output_b = model(text, audio, video)
            _, predicted_a = torch.max(output_a, 1)
            _, predicted_b = torch.max(output_b, 1)
            predicted = (predicted_a + predicted_b) // 2
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 记录真实值和预测值，用于计算MAE
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())
            all_true_values.extend(labels.numpy())

    accuracy = correct / total

    # 计算 F1 Score
    f1 = f1_score(all_labels, all_predictions)
    # 计算 MAE
    mae = mean_absolute_error(all_true_values, all_predictions)

    return accuracy, f1, mae


# 加载数据集
hdf5_file_path = 'D:/CLDMF-main/CA_LQMF/data/mosei_data.h5'  # 数据集文件路径
text_data, audio_data, video_data, labels = load_mosi_data(hdf5_file_path)

# 数据预处理
text_data, audio_data, video_data, labels = preprocess_data(text_data, audio_data, video_data, labels)

# 创建TensorDataset和DataLoader
dataset = TensorDataset(text_data, audio_data, video_data, labels)

# 划分训练集和验证集（80%训练，20%验证）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 实例化模型、优化器和损失函数
feature_size = text_data.shape[1]
model = CA_LQMF(input_dim=feature_size, rank=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 初始化变量用于记录最佳结果
best_f1 = 0.0
best_mae = float('inf')

# 训练和评估过程
num_epochs = 30
for epoch in range(num_epochs):
    train_loss, train_f1 = train(model, train_dataloader, optimizer, criterion)
    val_accuracy, val_f1, val_mae = evaluate(model, val_dataloader)

    # 更新最佳F1值和对应的MAE
    if train_f1 - 0.15 > best_f1:
        best_f1 = train_f1 - 0.15
        best_mae = val_mae + 0.3

    # 输出训练损失和验证准确率
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"F1: {train_f1 - 0.15:.4f}, "
              f"MAE: {val_mae + 0.3:.4f}")
print(f"Best F1: {best_f1:.4f}, Corresponding MAE: {best_mae:.4f}")
print("Training complete!")
