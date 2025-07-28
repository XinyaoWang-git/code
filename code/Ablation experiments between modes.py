import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import argparse
import json

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)


class MOSEIDataset(Dataset):
    """CMU-MOSEI数据集加载器"""

    def __init__(self, h5_path, split='train', available_modalities=None, debug=False):
        self.h5_path = h5_path
        self.split = split
        self.available_modalities = available_modalities
        self.debug = debug

        # 打开HDF5文件
        self.h5_file = h5py.File(h5_path, 'r')

        # 打印HDF5文件基本信息
        if self.debug:
            print(f"\n=== HDF5文件信息: {h5_path} ===")
            print(f"文件大小: {os.path.getsize(h5_path) / (1024 * 1024):.2f} MB")
            print(f"根目录包含: {list(self.h5_file.keys())}")

        # 获取数据集路径
        self._detect_dataset_paths()

        # 加载数据
        try:
            # 加载所有数据
            text = self.h5_file[self.text_path][:]
            audio = self.h5_file[self.audio_path][:]
            video = self.h5_file[self.video_path][:]
            labels = self.h5_file[self.labels_path][:].squeeze()

            # 根据split进行数据划分
            total_samples = len(labels)
            train_size = 16326
            valid_size = 1871
            test_size = 4659

            if split == 'train':
                start_idx, end_idx = 0, train_size
            elif split == 'valid':
                start_idx, end_idx = train_size, train_size + valid_size
            elif split == 'test':
                start_idx, end_idx = train_size + valid_size, train_size + valid_size + test_size
            else:
                raise ValueError(f"不支持的split类型: {split}")

            self.text = text[start_idx:end_idx]
            self.audio = audio[start_idx:end_idx]
            self.video = video[start_idx:end_idx]
            self.labels = labels[start_idx:end_idx]

            print(f"成功加载{split}数据集:")
            print(f"  文本特征: {self.text.shape}")
            print(f"  音频特征: {self.audio.shape}")
            print(f"  视频特征: {self.video.shape}")
            print(f"  标签: {self.labels.shape}")

            # 打印标签分布
            if debug and split == 'train':
                pos_count = np.sum(labels[start_idx:end_idx] > 0)
                neg_count = len(labels[start_idx:end_idx]) - pos_count
                print(f"  标签分布: 正样本 {pos_count}, 负样本 {neg_count}")

        except Exception as e:
            print(f"加载数据时出错: {e}")
            print(f"尝试访问的路径:")
            print(f"  文本: {self.text_path}")
            print(f"  音频: {self.audio_path}")
            print(f"  视频: {self.video_path}")
            print(f"  标签: {self.labels_path}")
            raise

    def _detect_dataset_paths(self):
        """自动检测数据集中的路径"""
        # 尝试直接访问根目录下的路径
        expected_paths = {
            'text': 'text',
            'audio': 'audio',
            'video': 'video',
            'labels': 'labels'
        }

        # 检查路径是否存在
        for key, path in expected_paths.items():
            if path not in self.h5_file:
                print(f"警告: 路径 '{path}' 不存在")

                # 尝试查找类似的路径
                similar_paths = [p for p in self.h5_file if key in p.lower()]
                if similar_paths:
                    print(f"找到类似路径: {similar_paths}")
                    expected_paths[key] = similar_paths[0]
                else:
                    raise ValueError(f"找不到与 '{key}' 相关的数据")

        self.text_path = expected_paths['text']
        self.audio_path = expected_paths['audio']
        self.video_path = expected_paths['video']
        self.labels_path = expected_paths['labels']

        if self.debug:
            print(f"数据集路径:")
            print(f"  文本: {self.text_path}")
            print(f"  音频: {self.audio_path}")
            print(f"  视频: {self.video_path}")
            print(f"  标签: {self.labels_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 默认所有模态都可用
        text = torch.FloatTensor(self.text[idx])
        audio = torch.FloatTensor(self.audio[idx])
        video = torch.FloatTensor(self.video[idx])
        label = torch.FloatTensor([self.labels[idx]])

        # 掩码，指示哪些模态可用
        mask = torch.ones(3)  # [text, audio, video]

        # 如果指定了可用模态，则将不可用的模态置为0
        if self.available_modalities:
            if 'l' not in self.available_modalities:
                text = torch.zeros_like(text)
                mask[0] = 0
            if 'a' not in self.available_modalities:
                audio = torch.zeros_like(audio)
                mask[1] = 0
            if 'v' not in self.available_modalities:
                video = torch.zeros_like(video)
                mask[2] = 0

        return {
            'text': text,
            'audio': audio,
            'video': video,
            'label': label,
            'mask': mask
        }


# 模态补偿自适应层
class AdaptiveCompensationLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AdaptiveCompensationLayer, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, text, audio, video, mask):
        # 连接所有模态
        combined = torch.cat([text, audio, video], dim=1)
        # 计算自适应权重
        weights = self.gate(combined)

        # 应用掩码，确保缺失模态的权重为0
        weights = weights * mask

        # 补偿操作
        text_compensated = text + weights[:, 0:1] * (audio + video) / 2
        audio_compensated = audio + weights[:, 1:2] * (text + video) / 2
        video_compensated = video + weights[:, 2:3] * (text + audio) / 2

        return text_compensated, audio_compensated, video_compensated


# 交叉注意力层 - 改进版本
# 交叉注意力层 - 修复版本
# 交叉注意力层 - 修复版本
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(CrossModalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 确保维度可以被均匀分割
        assert hidden_dim % num_heads == 0, "隐藏维度必须能被头数整除"

        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, query, key_value):
        batch_size = query.size(0)

        # 重塑为序列形式 [batch_size, seq_len, hidden_dim] -> [seq_len, batch_size, hidden_dim]
        query = query.unsqueeze(0)  # [1, batch_size, hidden_dim]

        # 检查key_value的来源模态数量
        num_modalities = key_value.size(0) // batch_size

        # 根据模态数量调整key_value的形状
        key_value = key_value.view(num_modalities, batch_size, self.hidden_dim)

        # 应用多头注意力
        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        # 重塑回批处理形式
        return attn_output.squeeze(0)  # [batch_size, hidden_dim]


# 动态集成模块
class DynamicIntegrationModule(nn.Module):
    def __init__(self, hidden_dim, num_modalities=3):
        super(DynamicIntegrationModule, self).__init__()
        self.modality_weights = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, num_modalities),
            nn.Softmax(dim=1)
        )

    def forward(self, text, audio, video, mask):
        # 连接所有模态
        combined = torch.cat([text, audio, video], dim=1)

        # 计算动态权重
        weights = self.modality_weights(combined)

        # 应用掩码，确保缺失模态的权重为0
        weights = weights * mask

        # 归一化权重
        available_modalities = mask.sum(dim=1, keepdim=True)
        if available_modalities.min() > 0:
            weights = weights / available_modalities
        else:
            # 如果所有模态都缺失，则返回零向量
            return torch.zeros_like(text)

        # 动态集成
        integrated = weights[:, 0:1] * text + weights[:, 1:2] * audio + weights[:, 2:3] * video

        return integrated


# CA-LQMDF模型
# CA-LQMDF模型
class CA_LQMDF(nn.Module):
    def __init__(self, feature_dim=300, hidden_dim=40, cross_layers=4, ablation=None):
        super(CA_LQMDF, self).__init__()
        self.ablation = ablation
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 特征投影层
        self.text_proj = nn.Linear(feature_dim, hidden_dim)
        self.audio_proj = nn.Linear(feature_dim, hidden_dim)
        self.video_proj = nn.Linear(feature_dim, hidden_dim)

        # 模态补偿自适应层
        if not (self.ablation and 'w/o A' in self.ablation):
            self.adaptive = AdaptiveCompensationLayer(hidden_dim)

        # 交叉注意力层
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim) for _ in range(cross_layers)
        ])

        # 动态集成模块
        if not (self.ablation and 'w/o D' in self.ablation):
            self.dynamic_integration = DynamicIntegrationModule(hidden_dim)
        else:
            # 简单连接后的投影层
            self.projection = nn.Linear(hidden_dim * 3, hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text, audio, video, mask):
        batch_size = text.size(0)

        # 特征投影
        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        video = self.video_proj(video)

        # 模态补偿
        if not (self.ablation and 'w/o A' in self.ablation):
            text, audio, video = self.adaptive(text, audio, video, mask)

        # 交叉注意力
        if self.ablation and 'w/o C' in self.ablation:
            # 使用自注意力代替交叉注意力
            for layer in self.cross_attn_layers:
                text = layer(text, text)
                audio = layer(audio, audio)
                video = layer(video, video)
        else:
            # 交叉注意力机制
            available_modalities = []
            modality_names = []

            if mask[0, 0] > 0:
                available_modalities.append(text)
                modality_names.append('text')
            if mask[0, 1] > 0:
                available_modalities.append(audio)
                modality_names.append('audio')
            if mask[0, 2] > 0:
                available_modalities.append(video)
                modality_names.append('video')

            # 确保至少有一个可用模态
            if available_modalities:
                # 构建key-value张量，将所有可用模态连接在一起
                key_value = torch.cat(available_modalities, dim=0)  # [batch_size*num_modalities, hidden_dim]

                for layer in self.cross_attn_layers:
                    if mask[0, 0] > 0:
                        text_cross = layer(text, key_value)
                        text = text + text_cross
                    if mask[0, 1] > 0:
                        audio_cross = layer(audio, key_value)
                        audio = audio + audio_cross
                    if mask[0, 2] > 0:
                        video_cross = layer(video, key_value)
                        video = video + video_cross

        # 特征融合
        if self.ablation and 'w/o D' in self.ablation:
            # 使用简单连接
            fused = torch.cat([text * mask[:, 0:1], audio * mask[:, 1:2], video * mask[:, 2:3]], dim=1)
            # 投影到隐藏维度
            fused = self.projection(fused)
        else:
            # 动态集成
            fused = self.dynamic_integration(text, audio, video, mask)

        # 分类预测
        output = self.classifier(fused)

        return output


# 训练模型
# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30, device='cuda'):
    model.to(device)
    best_val_loss = float('inf')
    best_model = None

    # 检查标签分布
    check_label_distribution(train_loader, device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            label = batch['label'].to(device)
            mask = batch['mask'].to(device)

            # 打印样本形状进行调试
            if batch_idx == 0 and epoch == 0:
                print(f"样本形状: text={text.shape}, audio={audio.shape}, video={video.shape}, label={label.shape}")
                print(f"可用模态: text={mask[0, 0]}, audio={mask[0, 1]}, video={mask[0, 2]}")

            optimizer.zero_grad()
            output = model(text, audio, video, mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            predicted = (output > 0).float()
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # 每100个batch打印一次训练信息
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")

        # 验证
        val_loss = evaluate_model(model, val_loader, criterion, device)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {100 * correct / total:.2f}%, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

    return best_model, best_val_loss


# 检查标签分布
def check_label_distribution(data_loader, device):
    pos_count = 0
    neg_count = 0
    total = 0

    for batch in data_loader:
        label = batch['label'].to(device)
        pos_count += (label > 0).sum().item()
        neg_count += (label <= 0).sum().item()
        total += label.size(0)

    print(
        f"数据标签分布: 正样本 {pos_count} ({100 * pos_count / total:.2f}%), 负样本 {neg_count} ({100 * neg_count / total:.2f}%)")


# 评估模型
def evaluate_model(model, data_loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            label = batch['label'].to(device)
            mask = batch['mask'].to(device)

            output = model(text, audio, video, mask)
            loss = criterion(output, label)
            total_loss += loss.item()

            # 计算准确率
            predicted = (output > 0).float()
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    print(f"评估结果: Loss={total_loss / len(data_loader):.4f}, Acc={accuracy:.2f}%")
    return total_loss / len(data_loader)


# 运行消融实验
def run_ablation_experiment(h5_path, available_modalities_list, ablation_types, batch_size=32, epochs=30,
                            device='cuda'):
    """运行消融实验"""
    results = {}

    # 对每种消融类型进行实验
    for ablation in ablation_types:
        print(f"\n=== 运行消融实验: {ablation if ablation else '完整模型'} ===")
        ablation_results = {}

        # 对每种可用模态组合进行实验
        for modalities in available_modalities_list:
            # 将集合符号转换为字符串表示
            modality_str = str(modalities)
            print(f"\n--- 可用模态: {modality_str} ---")

            try:
                # 加载数据集
                train_dataset = MOSEIDataset(h5_path, split='train', available_modalities=modalities, debug=True)
                val_dataset = MOSEIDataset(h5_path, split='valid', available_modalities=modalities)
                test_dataset = MOSEIDataset(h5_path, split='test', available_modalities=modalities)

                # 创建数据加载器
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)

                # 获取特征维度
                sample = train_dataset[0]
                feature_dim = sample['text'].shape[0]

                # 初始化模型
                model = CA_LQMDF(
                    feature_dim=feature_dim,  # 使用实际特征维度
                    hidden_dim=40,  # 设置隐藏维度为40
                    cross_layers=4,  # 设置交叉注意层数为4
                    ablation=ablation
                )

                # 打印模型结构
                print(model)

                # 定义损失函数和优化器
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)  # 设置学习率为0.001

                # 训练模型
                print(f"开始训练模型 - 消融: {ablation}, 可用模态: {modality_str}")
                best_model, best_val_loss = train_model(
                    model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device
                )

                # 加载最佳模型并评估
                model.load_state_dict(best_model)
                print(f"在测试集上评估模型 - 消融: {ablation}, 可用模态: {modality_str}")
                test_loss = evaluate_model(model, test_loader, criterion, device=device)

                # 计算准确率（二分类任务）
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in test_loader:
                        text = batch['text'].to(device)
                        audio = batch['audio'].to(device)
                        video = batch['video'].to(device)
                        label = batch['label'].to(device)
                        mask = batch['mask'].to(device)

                        output = model(text, audio, video, mask)
                        predicted = (output > 0).float()
                        total += label.size(0)
                        correct += (predicted == label).sum().item()

                accuracy = 100 * correct / total
                print(f"可用模态 {modality_str} - 测试准确率: {accuracy:.2f}%")
                ablation_results[modality_str] = accuracy

            except Exception as e:
                print(f"实验出错: {e}")
                import traceback
                traceback.print_exc()
                ablation_results[modality_str] = float('nan')
                continue

        # 计算平均准确率（不包括Avg本身）
        base_modalities = [str(m) for m in available_modalities_list if m != 'Avg']
        valid_scores = [ablation_results[m] for m in base_modalities if not np.isnan(ablation_results[m])]
        avg_accuracy = sum(valid_scores) / len(valid_scores) if valid_scores else float('nan')
        ablation_results['Avg'] = avg_accuracy
        results[ablation if ablation else '完整模型'] = ablation_results

    # 打印结果表格
    print("\n=== 消融实验结果 ===")
    # 确保Avg在最后一列
    modalities_with_avg = [str(m) for m in available_modalities_list if m != 'Avg'] + ['Avg']

    # 生成表头行
    header_cells = [f"{p}" for p in modalities_with_avg]
    header_line = "模型\t" + "\t".join(header_cells)
    print(header_line)

    # 生成数据行
    for ablation, scores in results.items():
        data_cells = [f"{scores[p]:.2f}" if not np.isnan(scores.get(p, float('nan'))) else "N/A" for p in
                      modalities_with_avg]
        data_line = f"{ablation}\t" + "\t".join(data_cells)
        print(data_line)

    # 返回结果字典
    return results


def main():
    # 设置参数
    parser = argparse.ArgumentParser(description='CA-LQMDF模型消融实验')
    parser.add_argument('--h5_path', type=str, default=r"E:\多模态数据融合\数据集\mosei_data.h5",
                        help='HDF5数据集文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()

    # 定义可用模态组合
    available_modalities_list = [
        {'l'},  # 只有文本
        {'a'},  # 只有音频
        {'v'},  # 只有视频
        {'l', 'a'},  # 文本和音频
        {'l', 'v'},  # 文本和视频
        {'a', 'v'},  # 音频和视频
        {'l', 'a', 'v'}  # 所有模态
    ]

    # 定义消融类型
    ablation_types = [
        None,  # 完整模型
        'w/o A',  # 无自适应补偿层
        'w/o C',  # 无交叉注意力
        'w/o D'  # 无动态集成
    ]

    # 运行消融实验
    results = run_ablation_experiment(
        h5_path=args.h5_path,
        available_modalities_list=available_modalities_list,
        ablation_types=ablation_types,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )

    # 保存结果到CSV和JSON
    df = pd.DataFrame(results)
    df.to_csv('results/ablation_results.csv')
    print("\n结果已保存到 ablation_results.csv")


if __name__ == "__main__":
    main()