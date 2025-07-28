import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import itertools


# 设置随机种子确保可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


class MOSIDataset(Dataset):
    def __init__(self, hdf5_path, split='train', missing_rate=0.0, missing_modality=None, random_missing=False,
                 train_ratio=0.7, val_ratio=0.15):
        """
        加载多模态数据集，支持多种HDF5文件结构
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.missing_rate = missing_rate
        self.missing_modality = missing_modality
        self.random_missing = random_missing

        try:
            with h5py.File(hdf5_path, 'r') as f:
                print(f"HDF5文件顶层键: {list(f.keys())}")

                # 自动检测文件结构类型
                structure_type = self._detect_structure_type(f)
                print(f"检测到的文件结构类型: {structure_type}")

                # 根据结构类型加载数据
                if structure_type == 'type1':  # split/modality
                    self.text = np.array(f[f'{split}/text'])
                    self.audio = np.array(f[f'{split}/audio'])
                    self.video = np.array(f[f'{split}/video'])
                    self.labels = np.array(f[f'{split}/labels'])

                elif structure_type == 'type2':  # modality/split
                    self.text = np.array(f[f'text/{split}'])
                    self.audio = np.array(f[f'audio/{split}'])
                    self.video = np.array(f[f'video/{split}'])
                    self.labels = np.array(f[f'labels/{split}'])

                elif structure_type == 'type3':  # 单独的train/val/test组
                    split_group = f[split]
                    self.text = np.array(split_group['text'])
                    self.audio = np.array(split_group['audio'])
                    self.video = np.array(split_group['video'])
                    self.labels = np.array(split_group['labels'])

                elif structure_type == 'type4':  # 模态包含split属性
                    self.text = np.array(self._get_split_data(f['text'], split))
                    self.audio = np.array(self._get_split_data(f['audio'], split))
                    self.video = np.array(self._get_split_data(f['video'], split))
                    self.labels = np.array(self._get_split_data(f['labels'], split))

                elif structure_type == 'type5':  # 单数据集，手动划分
                    # 获取整个数据集
                    text_all = np.array(f['text'])
                    audio_all = np.array(f['audio'])
                    video_all = np.array(f['video'])
                    labels_all = np.array(f['labels'])

                    # 计算划分索引
                    total_samples = len(labels_all)
                    train_size = int(total_samples * train_ratio)
                    val_size = int(total_samples * val_ratio)

                    # 根据split选择数据
                    if split == 'train':
                        start_idx, end_idx = 0, train_size
                    elif split == 'val':
                        start_idx, end_idx = train_size, train_size + val_size
                    elif split == 'test':
                        start_idx, end_idx = train_size + val_size, total_samples
                    else:
                        raise ValueError(f"未知的split类型: {split}")

                    # 划分数据
                    self.text = text_all[start_idx:end_idx]
                    self.audio = audio_all[start_idx:end_idx]
                    self.video = video_all[start_idx:end_idx]
                    self.labels = labels_all[start_idx:end_idx]

                    print(f"手动划分{split}集: 样本索引 {start_idx}-{end_idx - 1}, 共{end_idx - start_idx}个样本")

                else:
                    raise ValueError(f"不支持的HDF5文件结构类型: {structure_type}")

                print(f"成功加载 {split} 数据集 - 样本数: {len(self.labels)}")

        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise

    def _detect_structure_type(self, h5_file):
        """自动检测HDF5文件结构类型"""
        top_level_keys = list(h5_file.keys())

        # 检查是否包含train/val/test作为顶层键
        if 'train' in top_level_keys and 'val' in top_level_keys and 'test' in top_level_keys:
            # 检查这些键是否是组，并且包含模态
            if isinstance(h5_file['train'], h5py.Group) and 'text' in h5_file['train']:
                return 'type3'  # 结构类型3: train/val/test组

        # 检查是否包含text/audio/video作为顶层键
        if 'text' in top_level_keys and 'audio' in top_level_keys and 'video' in top_level_keys:
            # 检查这些模态下是否包含train/val/test
            if 'train' in h5_file['text'] and 'val' in h5_file['text']:
                return 'type2'  # 结构类型2: modality/split

            # 检查是否有split属性
            if hasattr(h5_file['text'], 'attrs') and 'split' in h5_file['text'].attrs:
                return 'type4'  # 结构类型4: 模态包含split属性

            # 检查是否是单个数据集（无划分）
            if all(key in top_level_keys for key in ['text', 'audio', 'video', 'labels']):
                # 确保这些是数据集而不是组
                if all(isinstance(h5_file[key], h5py.Dataset) for key in ['text', 'audio', 'video', 'labels']):
                    # 检查它们的长度是否一致
                    if len(h5_file['text']) == len(h5_file['audio']) == len(h5_file['video']) == len(h5_file['labels']):
                        return 'type5'  # 结构类型5: 单数据集，需要手动划分

        # 检查是否有split/modality结构
        if f'train/text' in h5_file:
            return 'type1'  # 结构类型1: split/modality

        return 'unknown'  # 未知结构

    def _get_split_data(self, dataset, split):
        """从包含split属性的数据集获取特定划分的数据"""
        # 假设数据集是一个大数组，每个样本有一个split属性
        # 这里需要根据实际情况调整
        split_indices = []

        # 检查是否有split属性
        if 'split' in dataset.attrs:
            # 可能整个数据集属于一个split
            if dataset.attrs['split'] == split:
                return dataset[:]
        else:
            # 假设每个样本有一个split标签
            for i in range(len(dataset)):
                # 这里需要根据实际数据结构调整
                # 例如，可能需要访问dataset[i]['split']或其他方式
                pass

        if not split_indices:
            raise ValueError(f"无法在数据集中找到 {split} 划分的数据")

        return dataset[split_indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = torch.FloatTensor(self.text[idx])
        audio = torch.FloatTensor(self.audio[idx])
        video = torch.FloatTensor(self.video[idx])
        label = torch.LongTensor([self.labels[idx]])

        # 应用缺失
        if self.random_missing and self.missing_rate > 0:
            # 随机决定是否缺失
            if np.random.random() < self.missing_rate:
                # 随机选择要缺失的模态
                modalities = ['text', 'audio', 'video']
                if self.missing_modality is None:
                    # 随机选择一个模态
                    missing_modality = np.random.choice(modalities)
                else:
                    missing_modality = self.missing_modality

                # 将选定的模态置零
                if missing_modality == 'text':
                    text = torch.zeros_like(text)
                elif missing_modality == 'audio':
                    audio = torch.zeros_like(audio)
                elif missing_modality == 'video':
                    video = torch.zeros_like(video)

        return text, audio, video, label.squeeze()


# -------------------------- 模型组件定义 --------------------------
class AdaptiveCompensationLayer(nn.Module):
    """自适应补偿层（A模块）：对缺失模态进行特征补偿"""

    def __init__(self, input_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))  # 缩放因子
        self.shift = nn.Parameter(torch.zeros(input_dim))  # 偏移因子
        self.gate = nn.Sigmoid()

    def forward(self, x, mask=None):
        # 对输入特征进行缩放和偏移，模拟缺失模态的补偿
        # mask: 指示哪些特征是缺失的（0表示缺失，1表示存在）
        if mask is not None:
            # 只对缺失的特征进行补偿
            compensation = self.gate(self.scale) * x + self.shift
            return x * mask + compensation * (1 - mask)
        else:
            return self.gate(self.scale) * x + self.shift


class CrossModalAttention(nn.Module):
    """交叉注意力模块（C模块）：实现模态间的信息交互"""

    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x1, x2):
        # x1: query, x2: key/value
        attn_output, _ = self.multihead_attn(
            x1.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)

        # 残差连接和层归一化
        x1 = self.norm1(x1 + attn_output)

        # 前馈网络
        ffn_output = self.ffn(x1)
        x1 = self.norm2(x1 + ffn_output)

        return x1


class DynamicIntegrationModule(nn.Module):
    """动态集成模块（D模块）：自适应地整合多模态信息"""

    def __init__(self, feature_dim, num_modalities=3):
        super().__init__()
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])

    def forward(self, *modalities):
        # 计算每个模态的注意力权重
        weights = [self.attention_weights[i](modality) for i, modality in enumerate(modalities)]

        # 归一化权重
        weights = torch.cat(weights, dim=1)
        weights = torch.softmax(weights, dim=1)

        # 加权融合
        fused = torch.zeros_like(modalities[0])
        for i, modality in enumerate(modalities):
            fused += modality * weights[:, i:i + 1]

        return fused, weights


# -------------------------- 完整模型定义 --------------------------
class CMFusionModel(nn.Module):
    """多模态融合模型，支持消融实验"""

    def __init__(self, text_dim=300, audio_dim=300, video_dim=300, hidden_dim=128,
                 use_adaptive=True, use_cross_attention=True, use_dynamic=True):
        super().__init__()

        # 编码器
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        self.video_encoder = nn.Linear(video_dim, hidden_dim)

        # 模块开关
        self.use_adaptive = use_adaptive
        self.use_cross_attention = use_cross_attention
        self.use_dynamic = use_dynamic

        # 自适应补偿层（A）
        if self.use_adaptive:
            self.adaptive_text = AdaptiveCompensationLayer(hidden_dim)
            self.adaptive_audio = AdaptiveCompensationLayer(hidden_dim)
            self.adaptive_video = AdaptiveCompensationLayer(hidden_dim)

        # 交叉注意力模块（C）
        if self.use_cross_attention:
            self.cross_text_audio = CrossModalAttention(hidden_dim)
            self.cross_text_video = CrossModalAttention(hidden_dim)
            self.cross_audio_text = CrossModalAttention(hidden_dim)
            self.cross_audio_video = CrossModalAttention(hidden_dim)
            self.cross_video_text = CrossModalAttention(hidden_dim)
            self.cross_video_audio = CrossModalAttention(hidden_dim)

        # 动态集成模块（D）
        if self.use_dynamic:
            self.dynamic_integration = DynamicIntegrationModule(hidden_dim, 3)
        else:
            self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2)  # 二分类
        )

    def forward(self, text, audio, video):
        # 编码
        text_feat = self.text_encoder(text)
        audio_feat = self.audio_encoder(audio)
        video_feat = self.video_encoder(video)

        # 自适应补偿（A）
        if self.use_adaptive:
            text_feat = self.adaptive_text(text_feat)
            audio_feat = self.adaptive_audio(audio_feat)
            video_feat = self.adaptive_video(video_feat)

        # 交叉注意力（C）
        if self.use_cross_attention:
            text_feat_a = self.cross_text_audio(text_feat, audio_feat)
            text_feat_v = self.cross_text_video(text_feat, video_feat)
            text_feat = (text_feat + text_feat_a + text_feat_v) / 3

            audio_feat_t = self.cross_audio_text(audio_feat, text_feat)
            audio_feat_v = self.cross_audio_video(audio_feat, video_feat)
            audio_feat = (audio_feat + audio_feat_t + audio_feat_v) / 3

            video_feat_t = self.cross_video_text(video_feat, text_feat)
            video_feat_v = self.cross_video_audio(video_feat, audio_feat)
            video_feat = (video_feat + video_feat_t + video_feat_v) / 3

        # 特征融合
        if self.use_dynamic:
            # 动态集成（D）
            fused_feat, _ = self.dynamic_integration(text_feat, audio_feat, video_feat)
        else:
            # 简单拼接
            fused_feat = torch.cat([text_feat, audio_feat, video_feat], dim=1)
            fused_feat = self.fusion(fused_feat)

        # 分类
        output = self.classifier(fused_feat)
        return output


# -------------------------- 训练和评估函数 --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    best_val_f1 = 0.0
    best_model = None

    for epoch in range(epochs):
        # 训练模式
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]'):
            text, audio, video, labels = batch
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(text, audio, video)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # 验证模式
        val_f1 = evaluate_model(model, val_loader, device)

        print(
            f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val F1: {val_f1:.4f}')

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, best_val_f1


def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            text, audio, video, labels = batch
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)

            outputs = model(text, audio, video)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算F1分数
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return f1


def run_ablation_experiment(hdf5_path, missing_rates, device, batch_size=32, epochs=10):
    # 定义所有可能的模型配置（完整模型和消融模型）
    model_configs = {
        'CA-LQMDF': {'use_adaptive': True, 'use_cross_attention': True, 'use_dynamic': True},
        'w/o A': {'use_adaptive': False, 'use_cross_attention': True, 'use_dynamic': True},
        'w/o C': {'use_adaptive': True, 'use_cross_attention': False, 'use_dynamic': True},
        'w/o D': {'use_adaptive': True, 'use_cross_attention': True, 'use_dynamic': False},
    }

    # 存储实验结果
    results = {config_name: [] for config_name in model_configs}

    # 对每个缺失率进行实验
    for missing_rate in missing_rates:
        print(f"\n=== 缺失率: {missing_rate} ===")

        # 创建测试集（使用随机缺失）
        test_dataset = MOSIDataset(
            hdf5_path,
            split='test',
            missing_rate=missing_rate,
            random_missing=True
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 对每个模型配置进行实验
        for config_name, config in model_configs.items():
            print(f"\n训练模型: {config_name}")

            # 创建训练集和验证集（无缺失，因为训练阶段不模拟缺失）
            train_dataset = MOSIDataset(hdf5_path, split='train', missing_rate=0.0)
            val_dataset = MOSIDataset(hdf5_path, split='val', missing_rate=0.0)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 初始化模型
            model = CMFusionModel(
                use_adaptive=config['use_adaptive'],
                use_cross_attention=config['use_cross_attention'],
                use_dynamic=config['use_dynamic']
            ).to(device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 训练模型
            model, _ = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs)

            # 在测试集上评估
            test_f1 = evaluate_model(model, test_loader, device)
            print(f"{config_name} 在缺失率 {missing_rate} 下的 F1 分数: {test_f1:.4f}")

            # 记录结果
            results[config_name].append(test_f1)

    # 绘制结果图
    plot_results(missing_rates, results)

    return results


def plot_results(missing_rates, results):
    """绘制不同缺失率下各模型的性能对比图"""
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
    plt.figure(figsize=(10, 6))

    for config_name, scores in results.items():
        plt.plot(missing_rates, scores, marker='o', label=config_name)

    plt.xlabel('模态内特征缺失率')
    plt.ylabel('F1 分数')
    plt.title('不同模块在不同缺失率下的性能对比')
    plt.grid(True)
    plt.legend()
    plt.xticks(missing_rates)
    plt.tight_layout()

    # 保存图像
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/ablation_results.png')
    plt.show()
def main():
    # 设置参数
    hdf5_path = "E:/多模态数据融合/数据集/mosei_data.h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 缺失率范围
    batch_size = 32
    epochs = 10  # 为了快速测试，可以减少轮数

    # 运行消融实验
    results = run_ablation_experiment(
        hdf5_path=hdf5_path,
        missing_rates=missing_rates,
        device=device,
        batch_size=batch_size,
        epochs=epochs
    )

    # 打印最终结果
    print("\n=== 最终实验结果 ===")
    for config_name, scores in results.items():
        print(f"{config_name}:")
        for rate, score in zip(missing_rates, scores):
            print(f"  缺失率 {rate * 100:.0f}%: F1 = {score:.4f}")


if __name__ == "__main__":
    main()