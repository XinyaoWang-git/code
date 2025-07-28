import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 加载IEMOCAP数据集 - 修改了键名以匹配实际HDF5文件结构
def load_iemocap_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # 检查并打印所有可用的键，用于验证
        print("HDF5文件中的数据集列表：")
        for key in f.keys():
            print(f"- {key}")

        text_data = np.array(f['text'])  # 修改为实际键名
        audio_data = np.array(f['audio'])  # 修改为实际键名
        video_data = np.array(f['video'])  # 修改为实际键名
        labels = np.array(f['labels'])  # 修改为实际键名
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


# 引入缺失值
def introduce_missingness(text, audio, video, missing_rate):
    """随机在模态中引入缺失值"""
    batch_size = text.size(0)

    # 为每个样本随机选择要缺失的模态
    for i in range(batch_size):
        if np.random.random() < missing_rate:
            # 随机选择缺失的模态
            modality_to_miss = np.random.randint(0, 3)  # 0:text, 1:audio, 2:video
            if modality_to_miss == 0:
                text[i] = 0  # 缺失文本
            elif modality_to_miss == 1:
                audio[i] = 0  # 缺失音频
            else:
                video[i] = 0  # 缺失视频

    return text, audio, video


# 模型定义

# 1. CA-LQMDF(ours)
class CALQMDF(nn.Module):
    def __init__(self, input_dim=40, num_cross_layers=4):
        super(CALQMDF, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # 交叉注意力层
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=128, num_heads=8)
            for _ in range(num_cross_layers)
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_encoder(text)
        audio_repr = self.audio_encoder(audio)
        video_repr = self.video_encoder(video)

        # 交叉注意力
        for attn in self.cross_attn_layers:
            # 文本-音频
            text_audio, _ = attn(
                text_repr.unsqueeze(0),
                audio_repr.unsqueeze(0),
                audio_repr.unsqueeze(0)
            )
            text_repr = text_repr + text_audio.squeeze(0)

            # 音频-视频
            audio_video, _ = attn(
                audio_repr.unsqueeze(0),
                video_repr.unsqueeze(0),
                video_repr.unsqueeze(0)
            )
            audio_repr = audio_repr + audio_video.squeeze(0)

            # 视频-文本
            video_text, _ = attn(
                video_repr.unsqueeze(0),
                text_repr.unsqueeze(0),
                text_repr.unsqueeze(0)
            )
            video_repr = video_repr + video_text.squeeze(0)

        # 融合表示
        combined = torch.cat([text_repr, audio_repr, video_repr], dim=1)
        output = self.fusion(combined)

        return output


# 2. Self-MM
class SelfMM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=768, num_layers=6, num_heads=8, dropout=0.1):
        super(SelfMM, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Linear(input_dim, hidden_dim)
        self.audio_encoder = nn.Linear(input_dim, hidden_dim)
        self.video_encoder = nn.Linear(input_dim, hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 2)

        # 对比学习温度参数
        self.tau = 0.1

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_encoder(text)
        audio_repr = self.audio_encoder(audio)
        video_repr = self.video_encoder(video)

        # 拼接所有模态表示
        combined = torch.stack([text_repr, audio_repr, video_repr], dim=1)  # [B, 3, H]

        # Transformer处理
        transformed = self.transformer_encoder(combined)

        # 取平均池化
        pooled = torch.mean(transformed, dim=1)

        # 分类输出
        output = self.classifier(pooled)

        return output


# 3. CubeMLP
class CubeMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_layers=4):
        super(CubeMLP, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Linear(input_dim, hidden_dim)
        self.audio_encoder = nn.Linear(input_dim, hidden_dim)
        self.video_encoder = nn.Linear(input_dim, hidden_dim)

        # MLP块
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_encoder(text)
        audio_repr = self.audio_encoder(audio)
        video_repr = self.video_encoder(video)

        # 拼接表示
        combined = torch.cat([text_repr, audio_repr, video_repr], dim=1)

        # 通过MLP块
        for block in self.mlp_blocks:
            combined = block(combined)

        # 分类输出
        output = self.classifier(combined)

        return output


# 4. MCTN
class MCTN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512, num_layers=6, num_heads=8):
        super(MCTN, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Linear(input_dim, hidden_dim)
        self.audio_encoder = nn.Linear(input_dim, hidden_dim)
        self.video_encoder = nn.Linear(input_dim, hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_encoder(text).unsqueeze(1)  # [B, 1, H]
        audio_repr = self.audio_encoder(audio).unsqueeze(1)
        video_repr = self.video_encoder(video).unsqueeze(1)

        # 拼接所有模态表示
        src = torch.cat([text_repr, audio_repr, video_repr], dim=1)  # [B, 3, H]

        # 编码
        memory = self.encoder(src)

        # 解码（自回归）
        tgt = memory[:, 0:1, :]  # 以文本为目标
        output = self.decoder(tgt, memory)

        # 分类输出
        output = self.classifier(output.squeeze(1))

        return output


# 5. TransM
class TransM(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=768, num_layers=6, num_heads=12):
        super(TransM, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Linear(input_dim, hidden_dim)
        self.audio_encoder = nn.Linear(input_dim, hidden_dim)
        self.video_encoder = nn.Linear(input_dim, hidden_dim)

        # 可学习位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, 3, hidden_dim))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_encoder(text).unsqueeze(1)  # [B, 1, H]
        audio_repr = self.audio_encoder(audio).unsqueeze(1)
        video_repr = self.video_encoder(video).unsqueeze(1)

        # 拼接所有模态表示
        combined = torch.cat([text_repr, audio_repr, video_repr], dim=1)  # [B, 3, H]

        # 添加位置编码
        combined = combined + self.pos_encoding

        # Transformer处理
        transformed = self.transformer(combined)

        # 取平均池化
        pooled = torch.mean(transformed, dim=1)

        # 分类输出
        output = self.classifier(pooled)

        return output


# 6. SMIL
class SMIL(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=768, num_layers=6, num_heads=8, tau=5.0):
        super(SMIL, self).__init__()

        # 文本编码器
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=num_layers)

        # 音视频编码器
        av_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.av_encoder = nn.TransformerEncoder(av_encoder_layer, num_layers=num_layers)

        # 投影头（用于对比学习）
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.av_proj = nn.Linear(hidden_dim, hidden_dim)

        # 模态编码器
        self.text_embed = nn.Linear(input_dim, hidden_dim)
        self.audio_embed = nn.Linear(input_dim, hidden_dim)
        self.video_embed = nn.Linear(input_dim, hidden_dim)

        # 分类器
        self.classifier = nn.Linear(hidden_dim * 2, 2)

        # 温度参数
        self.tau = tau

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_embed(text).unsqueeze(1)  # [B, 1, H]
        audio_repr = self.audio_embed(audio).unsqueeze(1)
        video_repr = self.video_embed(video).unsqueeze(1)

        # 文本编码
        text_encoded = self.text_encoder(text_repr)

        # 音视频编码
        av_combined = torch.cat([audio_repr, video_repr], dim=1)  # [B, 2, H]
        av_encoded = self.av_encoder(av_combined)
        av_pooled = torch.mean(av_encoded, dim=1, keepdim=True)  # [B, 1, H]

        # 投影到对比学习空间
        text_proj = self.text_proj(text_encoded.squeeze(1))
        av_proj = self.av_proj(av_pooled.squeeze(1))

        # 拼接用于分类
        combined = torch.cat([text_encoded.squeeze(1), av_pooled.squeeze(1)], dim=1)

        # 分类输出
        output = self.classifier(combined)

        return output


# 7. GCNet
class GCNet(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super(GCNet, self).__init__()

        # 模态编码器
        self.text_encoder = nn.Linear(input_dim, hidden_dim)
        self.audio_encoder = nn.Linear(input_dim, hidden_dim)
        self.video_encoder = nn.Linear(input_dim, hidden_dim)

        # 图卷积层
        self.gcn = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text, audio, video):
        # 编码各模态
        text_repr = self.text_encoder(text)
        audio_repr = self.audio_encoder(audio)
        video_repr = self.video_encoder(video)

        # 构建图表示（简单拼接）
        graph_repr = torch.cat([text_repr, audio_repr, video_repr], dim=1)

        # 图卷积
        graph_output = self.gcn(graph_repr)

        # 分类输出
        output = self.classifier(graph_output)

        return output


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for text, audio, video, labels in train_loader:
        # 移至设备
        text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)

        # 引入缺失值
        text, audio, video = introduce_missingness(text, audio, video, 0.0)  # 训练时不引入缺失值

        # 前向传播
        optimizer.zero_grad()
        outputs = model(text, audio, video)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_loader), correct / total


# 评估函数
def evaluate_model(model, test_loader, criterion, device, missing_rate):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for text, audio, video, labels in test_loader:
            # 移至设备
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)

            # 引入缺失值
            text, audio, video = introduce_missingness(text, audio, video, missing_rate)

            # 前向传播
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 保存所有标签和预测结果用于F1计算
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算F1分数
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # 计算MAE（将分类问题转换为回归问题的MAE）
    mae = mean_absolute_error(all_labels, all_preds)

    return total_loss / len(test_loader), correct / total, f1, mae


# 主函数
def main():
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    data_path = "E:/多模态数据融合/数据集/iemocap_data.h5"  # 修改为实际数据路径
    text_data, audio_data, video_data, labels = load_iemocap_data(data_path)
    text_data, audio_data, video_data, labels = preprocess_data(text_data, audio_data, video_data, labels)

    # 创建数据集
    dataset = TensorDataset(text_data, audio_data, video_data, labels)

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 定义不同的缺失率
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # 定义所有模型 - 动态获取输入维度
    input_dim = text_data.shape[1]
    models = {
        "Self-MM": lambda: SelfMM(input_dim=input_dim, hidden_dim=768, num_layers=6, num_heads=8, dropout=0.1),
        "CubeMLP": lambda: CubeMLP(input_dim=input_dim, hidden_dim=256, num_layers=4),
        "MCTN": lambda: MCTN(input_dim=input_dim, hidden_dim=512, num_layers=6, num_heads=8),
        "TransM": lambda: TransM(input_dim=input_dim, hidden_dim=768, num_layers=6, num_heads=12),
        "SMIL": lambda: SMIL(input_dim=input_dim, hidden_dim=768, num_layers=6, num_heads=8, tau=5.0),
        "GCNet": lambda: GCNet(input_dim=input_dim, hidden_dim=128),
        "CA-LQMDF(ours)": lambda: CALQMDF(input_dim=input_dim, num_cross_layers=4)  # 修正输入维度
    }

    # 结果记录
    f1_results = {name: [] for name in models}
    mae_results = {name: [] for name in models}

    # 训练和评估每个模型
    for model_name, model_fn in models.items():
        print(f"\n=== 训练和评估 {model_name} ===")

        for missing_rate in missing_rates:
            print(f"\n--- 缺失率: {missing_rate} ---")

            # 初始化模型
            model = model_fn().to(device)

            # 定义优化器和损失函数
            if model_name == "MCTN":
                optimizer = optim.AdamW(model.parameters(), lr=0.001)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            criterion = nn.CrossEntropyLoss()

            # 训练模型
            num_epochs = 30
            best_val_f1 = 0.0
            best_model = None

            for epoch in range(num_epochs):
                # 训练
                train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

                # 验证
                val_loss, val_acc, val_f1, val_mae = evaluate_model(model, val_loader, criterion, device, missing_rate)

                # 保存最佳模型
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model = model.state_dict().copy()

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}: "
                          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

            # 加载最佳模型并在测试集上评估
            model.load_state_dict(best_model)
            test_loss, test_acc, test_f1, test_mae = evaluate_model(model, test_loader, criterion, device, missing_rate)

            print(f"测试结果: Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}, MAE={test_mae:.4f}")

            # 记录结果
            f1_results[model_name].append(test_f1)
            mae_results[model_name].append(test_mae)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    for model_name, values in f1_results.items():
        plt.plot(missing_rates, values, marker='o', label=model_name)

    plt.xlabel('缺失率')
    plt.ylabel('F1分数')
    plt.title('不同缺失率下各模型的F1分数对比')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iemocap_f1_scores.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    for model_name, values in mae_results.items():
        plt.plot(missing_rates, values, marker='o', label=model_name)

    plt.xlabel('缺失率')
    plt.ylabel('MAE')
    plt.title('不同缺失率下各模型的MAE对比')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iemocap_mae.png')
    plt.close()

    print("实验完成！结果已保存为图片文件。")


if __name__ == "__main__":
    main()
