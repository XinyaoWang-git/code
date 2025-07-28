import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_absolute_error
import os
import crossmodal
import dynamic
import adaptive
import matplotlib.pyplot as plt


# 加载单个H5文件
def load_single_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        text_data = np.array(f['text'])
        audio_data = np.array(f['audio'])
        video_data = np.array(f['video'])
        labels = np.array(f['labels'])
    return text_data, audio_data, video_data, labels


# 数据预处理
def preprocess_data(text_data, audio_data, video_data, labels):
    # 处理缺失值
    text_data = np.nan_to_num(text_data, nan=0.0)
    audio_data = np.nan_to_num(audio_data, nan=0.0)
    video_data = np.nan_to_num(video_data, nan=0.0)

    # 转换为PyTorch张量
    text_data = torch.tensor(text_data).float()
    audio_data = torch.tensor(audio_data).float()
    video_data = torch.tensor(video_data).float()
    labels = torch.tensor(labels, dtype=torch.long)

    # 确保标签范围在[0,1]
    labels = torch.clamp(labels, min=0, max=1)

    return text_data, audio_data, video_data, labels


# 信息蒸馏模块（用于CA-LQMF）
class SelfDistillationModule(nn.Module):
    def __init__(self, input_dim):
        super(SelfDistillationModule, self).__init__()
        self.fc_a = nn.Linear(input_dim, input_dim)
        self.fc_b = nn.Linear(input_dim, input_dim)

    def forward(self, h_a, h_b):
        distill_a = self.fc_a(h_a)
        distill_b = self.fc_b(h_b)
        return distill_a, distill_b


# 定义所有模型
class CA_LQMF(nn.Module):
    def __init__(self, input_dim, rank=4):
        super(CA_LQMF, self).__init__()
        # 假设crossmodal和dynamic模块已定义
        self.saca = crossmodal.CrossModalInteractionLayer(input_dim)
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


class SelfMM(nn.Module):
    def __init__(self, input_dim):
        super(SelfMM, self).__init__()
        self.fc_text = nn.Linear(input_dim, 128)
        self.fc_audio = nn.Linear(input_dim, 128)
        self.fc_video = nn.Linear(input_dim, 128)
        # 修改为正确的输入维度
        self.classifier = nn.Linear(128 * 3, 2)

    def forward(self, text, audio, video):
        text_repr = torch.relu(self.fc_text(text))
        audio_repr = torch.relu(self.fc_audio(audio))
        video_repr = torch.relu(self.fc_video(video))
        combined = torch.cat([text_repr, audio_repr, video_repr], dim=1)
        return self.classifier(combined)


class CubeMLP(nn.Module):
    def __init__(self, input_dim):
        super(CubeMLP, self).__init__()
        self.mlp_layers = nn.ModuleList([
            nn.Linear(input_dim * 3, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(256),
            nn.LayerNorm(128),
            nn.LayerNorm(64),
            nn.LayerNorm(32)
        ])
        self.classifier = nn.Linear(32, 2)

    def forward(self, text, audio, video):
        combined = torch.cat([text, audio, video], dim=1)
        x = combined
        for mlp, norm in zip(self.mlp_layers, self.norm_layers):
            x = torch.relu(mlp(x))
            x = norm(x)
        return self.classifier(x)


class MCTN(nn.Module):
    def __init__(self, input_dim):
        super(MCTN, self).__init__()
        self.fc_text = nn.Linear(input_dim, 512)
        self.fc_audio = nn.Linear(input_dim, 512)
        self.fc_video = nn.Linear(input_dim, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.classifier = nn.Linear(512, 2)

    def forward(self, text, audio, video):
        text_repr = self.fc_text(text).unsqueeze(1)
        audio_repr = self.fc_audio(audio).unsqueeze(1)
        video_repr = self.fc_video(video).unsqueeze(1)
        sequence = torch.cat([text_repr, audio_repr, video_repr], dim=1)
        encoded = self.encoder(sequence)
        decoded = self.decoder(encoded, encoded)
        pooled = torch.mean(decoded, dim=1)
        return self.classifier(pooled)


class TransM(nn.Module):
    def __init__(self, input_dim):
        super(TransM, self).__init__()
        self.fc_text = nn.Linear(input_dim, 768)
        self.fc_audio = nn.Linear(input_dim, 768)
        self.fc_video = nn.Linear(input_dim, 768)
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, 768))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=6
        )
        self.classifier = nn.Linear(768, 2)

    def forward(self, text, audio, video):
        text_repr = self.fc_text(text).unsqueeze(1)
        audio_repr = self.fc_audio(audio).unsqueeze(1)
        video_repr = self.fc_video(video).unsqueeze(1)
        sequence = torch.cat([text_repr, audio_repr, video_repr], dim=1)
        sequence += self.pos_encoding
        transformed = self.transformer(sequence)
        pooled = torch.mean(transformed, dim=1)
        return self.classifier(pooled)


class SMIL(nn.Module):
    def __init__(self, input_dim):
        super(SMIL, self).__init__()
        self.fc_text = nn.Linear(input_dim, 768)
        self.fc_audio = nn.Linear(input_dim, 768)
        self.fc_video = nn.Linear(input_dim, 768)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=6
        )
        self.audio_video_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=6
        )
        self.classifier = nn.Linear(768 * 2, 2)
        self.tau = 5

    def forward(self, text, audio, video):
        text_repr = self.fc_text(text).unsqueeze(1)
        audio_repr = self.fc_audio(audio).unsqueeze(1)
        video_repr = self.fc_video(video).unsqueeze(1)
        text_transformed = self.text_transformer(text_repr)
        audio_video = torch.cat([audio_repr, video_repr], dim=1)
        audio_video_transformed = self.audio_video_transformer(audio_video)
        pooled_text = torch.mean(text_transformed, dim=1)
        pooled_audio_video = torch.mean(audio_video_transformed, dim=1)
        combined = torch.cat([pooled_text, pooled_audio_video], dim=1)
        return self.classifier(combined)


class GCNet(nn.Module):
    def __init__(self, input_dim):
        super(GCNet, self).__init__()
        self.fc_text = nn.Linear(input_dim, 128)
        self.fc_audio = nn.Linear(input_dim, 128)
        self.fc_video = nn.Linear(input_dim, 128)
        self.gcn = nn.Linear(128 * 3, 128)
        self.classifier = nn.Linear(128, 2)

    def forward(self, text, audio, video):
        text_repr = torch.relu(self.fc_text(text))
        audio_repr = torch.relu(self.fc_audio(audio))
        video_repr = torch.relu(self.fc_video(video))
        graph_repr = torch.cat([text_repr, audio_repr, video_repr], dim=1)
        graph_output = torch.relu(self.gcn(graph_repr))
        return self.classifier(graph_output)


# 训练过程
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for batch in tqdm(dataloader, desc="Training"):
        text, audio, video, labels = batch
        optimizer.zero_grad()

        # 处理CA-LQMF的双输出
        if isinstance(model, CA_LQMF):
            output_a, output_b = model(text, audio, video)
            loss = criterion(output_a, labels) + criterion(output_b, labels)
            _, predicted = torch.max((output_a + output_b) / 2, 1)
        else:
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # 计算F1分数
    f1 = f1_score(all_labels, all_predictions)
    return running_loss / len(dataloader), f1


# 评估过程
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text, audio, video, labels = batch

            # 处理CA-LQMF的双输出
            if isinstance(model, CA_LQMF):
                output_a, output_b = model(text, audio, video)
                _, predicted = torch.max((output_a + output_b) / 2, 1)
            else:
                outputs = model(text, audio, video)
                _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_true_values.extend(labels.cpu().numpy())

    # 计算评估指标
    f1 = f1_score(all_labels, all_predictions)
    mae = mean_absolute_error(all_true_values, all_predictions)
    return f1, mae


# 主函数
def main():
    # 单个H5文件路径
    h5_file_path = 'E:\多模态数据融合\数据集\mosei_data.h5'

    # 加载数据
    text_data, audio_data, video_data, labels = load_single_h5_file(h5_file_path)
    text_data, audio_data, video_data, labels = preprocess_data(text_data, audio_data, video_data, labels)

    # 创建数据集
    dataset = TensorDataset(text_data, audio_data, video_data, labels)

    # 划分数据集
    train_size = 16326
    valid_size = 1871
    test_size = 4659
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # 定义不同的缺失率
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # 定义所有模型
    model_classes = {
        'Self-MM': SelfMM,
        'CubeMLP': CubeMLP,
        'MCTN': MCTN,
        'TransM': TransM,
        'SMIL': SMIL,
        'GCNet': GCNet,
        'CA-LQMF(ours)': CA_LQMF
    }

    # 记录所有结果
    f1_results = {model_name: [] for model_name in model_classes}
    mae_results = {model_name: [] for model_name in model_classes}

    # 特征维度
    feature_size = text_data.shape[1]

    # 对每个模型进行训练和评估
    for model_name, model_class in model_classes.items():
        print(f"\n=== Training and evaluating {model_name} ===")

        for rate in missing_rates:
            print(f"\nMissing rate: {rate}")

            # 引入缺失率
            def introduce_missing(data, rate):
                mask = torch.rand(data.shape) < rate
                return data.masked_fill(mask, 0)

            train_text, train_audio, train_video, train_labels = zip(
                *[train_dataset[i] for i in range(len(train_dataset))])
            train_text = torch.stack([introduce_missing(t, rate) for t in train_text])
            train_audio = torch.stack([introduce_missing(a, rate) for a in train_audio])
            train_video = torch.stack([introduce_missing(v, rate) for v in train_video])
            train_dataset_miss = TensorDataset(train_text, train_audio, train_video, torch.tensor(train_labels))
            train_dataloader = DataLoader(train_dataset_miss, batch_size=32, shuffle=True)

            val_text, val_audio, val_video, val_labels = zip(*[valid_dataset[i] for i in range(len(valid_dataset))])
            val_text = torch.stack([introduce_missing(t, rate) for t in val_text])
            val_audio = torch.stack([introduce_missing(a, rate) for a in val_audio])
            val_video = torch.stack([introduce_missing(v, rate) for v in val_video])
            val_dataset_miss = TensorDataset(val_text, val_audio, val_video, torch.tensor(val_labels))
            val_dataloader = DataLoader(val_dataset_miss, batch_size=32, shuffle=False)

            test_text, test_audio, test_video, test_labels = zip(*[test_dataset[i] for i in range(len(test_dataset))])
            test_text = torch.stack([introduce_missing(t, rate) for t in test_text])
            test_audio = torch.stack([introduce_missing(a, rate) for a in test_audio])
            test_video = torch.stack([introduce_missing(v, rate) for v in test_video])
            test_dataset_miss = TensorDataset(test_text, test_audio, test_video, torch.tensor(test_labels))
            test_dataloader = DataLoader(test_dataset_miss, batch_size=32, shuffle=False)

            # 初始化模型
            if model_name == 'CA-LQMF(ours)':
                model = model_class(input_dim=feature_size, rank=4)
            else:
                model = model_class(input_dim=feature_size)

            if model_name == 'MCTN':
                optimizer = optim.AdamW(model.parameters(), lr=0.001)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            criterion = nn.CrossEntropyLoss()

            # 训练模型
            best_f1 = 0.0
            best_mae = float('inf')
            num_epochs = 30

            for epoch in range(num_epochs):
                train_loss, train_f1 = train(model, train_dataloader, optimizer, criterion)
                val_f1, val_mae = evaluate(model, val_dataloader)

                # 保存最佳结果
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_mae = val_mae

                # 每5个epoch打印一次进度
                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}, Val MAE: {val_mae:.4f}")

            # 测试模型
            test_f1, test_mae = evaluate(model, test_dataloader)

            # 记录结果
            f1_results[model_name].append(test_f1)
            mae_results[model_name].append(test_mae)
            print(f"Best F1: {best_f1:.4f}, Best MAE: {best_mae:.4f}, Test F1: {test_f1:.4f}, Test MAE: {test_mae:.4f}")

    # 可视化结果
    # F1分数可视化
    plt.figure(figsize=(10, 6))
    for model_name, f1_values in f1_results.items():
        plt.plot(missing_rates, f1_values, marker='o', label=model_name)
    plt.xlabel('Missing Rate')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Missing Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('F1_mosei.png')
    plt.close()

    # MAE可视化
    plt.figure(figsize=(10, 6))
    for model_name, mae_values in mae_results.items():
        plt.plot(missing_rates, mae_values, marker='o', label=model_name)
    plt.xlabel('Missing Rate')
    plt.ylabel('MAE')
    plt.title('MAE vs Missing Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('MAE_mosei.png')
    plt.close()


if __name__ == "__main__":
    main()
