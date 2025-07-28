import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2
from sklearn.metrics import accuracy_score, f1_score

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 1. 自定义数据集类（修复标签尺寸不匹配和越界问题）
class NYUDepthV2(Dataset):
    """NYU Depth V2标记数据集加载器（修复标签问题）"""

    def __init__(self, root_dir, split='train', transform=None, train_ratio=0.8, debug=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.cache = {}  # 缓存已读取的数据
        self.debug = debug  # 调试模式，用于验证标签
        self.invalid_labels = set()  # 记录已发现的无效标签值

        # 验证标记数据集文件
        self.mat_file = os.path.join(root_dir, 'nyu_depth_v2_labeled.mat')
        if not os.path.exists(self.mat_file):
            raise FileNotFoundError(f"未找到标记数据集: {self.mat_file}")

        # 预计算训练/测试划分
        with h5py.File(self.mat_file, 'r') as f:
            total_samples = len(f['images'])  # 1449个标记样本
            indices = np.arange(total_samples)
            np.random.shuffle(indices)
            train_size = int(total_samples * train_ratio)

            self.indices = indices[:train_size] if split == 'train' else indices[train_size:]

            # 数据验证（仅首次运行时执行）
            if not os.path.exists(os.path.join(root_dir, '.data_verified')):
                self._verify_data(f)
                with open(os.path.join(root_dir, '.data_verified'), 'w') as vf:
                    vf.write('Data verified')

    def _verify_data(self, h5_file):
        """验证数据集标签范围"""
        print("正在验证数据集标签范围...")
        max_label = 0
        for i in tqdm(range(len(self.indices))):
            idx = self.indices[i]
            label = np.array(h5_file['labels'][idx]).transpose(1, 0)
            current_max = np.max(label)
            if current_max > max_label:
                max_label = current_max
        print(f"标签最大值: {max_label}")
        if max_label > 39:
            print("警告: 发现超出40类(0-39)的标签值，将在加载时自动裁剪")
        else:
            print("标签范围正常(0-39)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # 缓存命中则直接返回
        if actual_idx in self.cache:
            img, depth, label = self.cache[actual_idx]
        else:
            # 每次读取时单独打开文件
            with h5py.File(self.mat_file, 'r') as f:
                # 读取RGB图像（转置维度适配OpenCV）
                img = np.array(f['images'][actual_idx]).transpose(2, 1, 0)  # (3,480,640)→(480,640,3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转为BGR格式

                # 读取深度图（转置维度）
                depth = np.array(f['depths'][actual_idx]).transpose(1, 0)  # (480,640)

                # 读取语义标签（转置维度）
                label = np.array(f['labels'][actual_idx]).transpose(1, 0)  # (480,640)

                # 关键修复：过滤无效标签（NYU Depth V2标签应为0-39）
                original_max = np.max(label)
                label = np.clip(label, 0, 39)  # 将超出0-39的标签强制截断

                # 调试信息（首次发现无效标签时输出）
                if self.debug and original_max > 39 and original_max not in self.invalid_labels:
                    print(f"样本 {actual_idx} 存在无效标签值 {original_max}，已裁剪到0-39")
                    self.invalid_labels.add(original_max)  # 记录已发现的无效标签值

                # 缓存数据
                if len(self.cache) < 200:  # 最多缓存200个样本
                    self.cache[actual_idx] = (img.copy(), depth.copy(), label.copy())

        # 应用数据变换（确保标签尺寸与输入一致）
        if self.transform:
            img = self.transform(img)  # 应用完整变换（包括Resize到256x256）

            # 对深度图单独处理（确保尺寸一致）
            depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
            depth = transforms.Resize((256, 256))(depth)  # 缩放到256x256

            # 对标签单独处理（使用最近邻插值，避免类别混淆）
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
            label = torch.tensor(label, dtype=torch.long)

        return img, depth, label


# 2. 自适应层实现
class AdaptiveLayer(nn.Module):
    def __init__(self, in_features, tech='scale_shift'):
        super().__init__()
        self.tech = tech
        self.in_features = in_features

        if tech == 'scale_only':
            self.scale = nn.Parameter(torch.ones(1, in_features, 1, 1))
        elif tech == 'shift_only':
            self.shift = nn.Parameter(torch.zeros(1, in_features, 1, 1))
        elif tech == 'scale_shift':
            self.scale = nn.Parameter(torch.ones(1, in_features, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, in_features, 1, 1))
        elif tech == 'bitfit':
            self.bias = nn.Parameter(torch.zeros(1, in_features, 1, 1))
        elif tech == 'lora':
            self.rank = 8
            self.A = nn.Parameter(torch.randn(in_features, self.rank) * 0.01)
            self.B = nn.Parameter(torch.zeros(self.rank, in_features))
        elif tech in ['pretrained', 'dedicated', 'norm']:
            pass  # 无额外参数
        else:
            raise ValueError(f"不支持的自适应技术: {tech}")

    def forward(self, x):
        if self.tech == 'scale_only':
            return x * self.scale
        elif self.tech == 'shift_only':
            return x + self.shift
        elif self.tech == 'scale_shift':
            return x * self.scale + self.shift
        elif self.tech == 'bitfit':
            return x + self.bias
        elif self.tech == 'lora':
            batch_size, channels, height, width = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
            lora_update = (x_flat @ self.A @ self.B).reshape(batch_size, height, width, channels)
            return x + lora_update.permute(0, 3, 1, 2)
        elif self.tech == 'norm':
            return nn.functional.layer_norm(x, x.shape[1:])
        else:
            return x


# 3. 带自适应层的分割模型
class AdaptiveSegmentationModel(nn.Module):
    def __init__(self, adaptive_tech, modality='rgb'):
        super().__init__()
        self.backbone = self._build_backbone(adaptive_tech, modality)

        # 冻结预训练权重（除dedicated外）
        if adaptive_tech != 'dedicated':
            for param in self.backbone.encoder.parameters():
                param.requires_grad = False

        # 编码器各阶段输出维度（ResNet18）
        self.adaptive_layers = nn.ModuleList([
            AdaptiveLayer(feat_dim, adaptive_tech)
            for feat_dim in [64, 128, 256, 512]
        ])

    def _build_backbone(self, adaptive_tech, modality):
        from torchvision.models import resnet18
        import torch.nn.functional as F

        # 编码器：ResNet18
        encoder = resnet18(pretrained=(adaptive_tech == 'pretrained'))
        # 适配输入通道（RGB=3，深度图=1）
        in_channels = 3 if modality == 'rgb' else 1
        if in_channels != 3:
            encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 解码器：简单上采样模块
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.final_conv = nn.Conv2d(32, 40, 1)  # 40类输出

            def forward(self, features):
                x = features[3]  # 512维特征
                x = self.up1(x) + features[2]  # 与256维特征融合
                x = self.up2(x) + features[1]  # 与128维特征融合
                x = self.up3(x) + features[0]  # 与64维特征融合
                x = self.up4(x)
                return self.final_conv(F.interpolate(x, size=(256, 256), mode='bilinear'))

        # 组合编码器和解码器
        class UNet(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                # 提取编码器特征
                x1 = self.encoder.conv1(x)
                x1 = self.encoder.bn1(x1)
                x1 = self.encoder.relu(x1)
                x1 = self.encoder.maxpool(x1)  # 64维

                x2 = self.encoder.layer1(x1)  # 64维
                x3 = self.encoder.layer2(x2)  # 128维
                x4 = self.encoder.layer3(x3)  # 256维
                x5 = self.encoder.layer4(x4)  # 512维
                return [x2, x3, x4, x5]

            def decode(self, features):
                return self.decoder(features)

        return UNet(encoder, Decoder())

    def forward(self, x):
        features = self.backbone(x)
        adapted_features = [self.adaptive_layers[i](f) for i, f in enumerate(features)]
        return self.backbone.decode(adapted_features)


# 4. 评估指标计算
def compute_metrics(pred, mask):
    pred = pred.argmax(dim=1).flatten().cpu().numpy()
    mask = mask.flatten().cpu().numpy()

    valid = mask != 0  # 忽略背景类
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    pred, mask = pred[valid], mask[valid]

    macc = accuracy_score(mask, pred) * 100
    f1 = f1_score(mask, pred, average='macro') * 100

    # 计算mIoU
    classes = np.unique(np.concatenate([mask, pred]))
    ious = []
    for cls in classes:
        if cls == 0:
            continue
        intersection = (pred == cls) & (mask == cls)
        union = (pred == cls) | (mask == cls)
        ious.append(intersection.sum() / union.sum() if union.sum() > 0 else 0)
    miou = np.mean(ious) * 100 if ious else 0.0

    return macc, f1, miou


# 5. 训练与评估
def train_and_evaluate(adaptive_tech, modality, epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 技术: {adaptive_tech} | 模态: {modality}")

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # 输入图像缩放到256x256
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406] if modality == 'rgb' else [0.5],
            std=[0.229, 0.224, 0.225] if modality == 'rgb' else [0.5]
        )
    ])

    # 加载数据集（单线程读取，启用调试模式）
    dataset_path = r"E:\多模态数据融合\源代码\NYU Depth V2"
    train_dataset = NYUDepthV2(dataset_path, split='train', transform=transform, debug=True)
    test_dataset = NYUDepthV2(dataset_path, split='test', transform=transform, debug=False)

    # 单线程DataLoader（避免HDF5冲突）
    batch_size = 4 if device.type == 'cpu' else 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 关键：单线程
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 模型初始化
    model = AdaptiveSegmentationModel(adaptive_tech, modality).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练过程
    best_miou = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # 训练
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for rgb, depth, mask in progress_bar:
            x = rgb.to(device) if modality == 'rgb' else depth.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证
        model.eval()
        macc_list, f1_list, miou_list = [], [], []
        with torch.no_grad():
            for rgb, depth, mask in test_loader:
                x = rgb.to(device) if modality == 'rgb' else depth.to(device)
                output = model(x)
                macc, f1, miou = compute_metrics(output, mask)
                macc_list.append(macc)
                f1_list.append(f1)
                miou_list.append(miou)

        avg_macc = np.mean(macc_list)
        avg_f1 = np.mean(f1_list)
        avg_miou = np.mean(miou_list)
        print(f"Epoch {epoch + 1} | 平均损失: {train_loss / len(train_loader):.4f} | "
              f"mAcc: {avg_macc:.2f} | F1: {avg_f1:.2f} | mIoU: {avg_miou:.2f}")

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), f"best_model_{adaptive_tech}_{modality}.pth")

    return {'mAcc': avg_macc, 'F1': avg_f1, 'mIoU': avg_miou}


# 6. 主实验流程
if __name__ == "__main__":
    adaptive_techs = [
        'pretrained', 'dedicated', 'scale_only', 'shift_only',
        'bitfit', 'lora', 'norm', 'scale_shift'
    ]
    modalities = ['rgb', 'depth']
    results = []

    for tech in adaptive_techs:
        tech_results = {}
        for modality in modalities:
            print(f"\n===== 实验: {tech} - {modality} =====")
            metrics = train_and_evaluate(tech, modality, epochs=10)
            tech_results[modality] = metrics

            results.append({
                '模型': tech,
                '模态': modality.upper(),
                'mAcc': metrics['mAcc'],
                'F1': metrics['F1'],
                'mIoU': metrics['mIoU']
            })

        # 计算平均性能
        avg_macc = (tech_results['rgb']['mAcc'] + tech_results['depth']['mAcc']) / 2
        avg_f1 = (tech_results['rgb']['F1'] + tech_results['depth']['F1']) / 2
        avg_miou = (tech_results['rgb']['mIoU'] + tech_results['depth']['mIoU']) / 2
        results.append({'模型': tech, '模态': 'Average', 'mAcc': avg_macc, 'F1': avg_f1, 'mIoU': avg_miou})

    # 保存结果
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='模型', columns='模态', values=['mAcc', 'F1', 'mIoU'])
    pivot_df = pivot_df.reindex(columns=[
        ('mAcc', 'RGB'), ('F1', 'RGB'), ('mIoU', 'RGB'),
        ('mAcc', 'Depth'), ('F1', 'Depth'), ('mIoU', 'Depth'),
        ('mAcc', 'Average'), ('F1', 'Average'), ('mIoU', 'Average')
    ])
    pivot_df.to_csv('adaptive_performance.csv')
    print("\n结果已保存至adaptive_performance.csv")
    print(pivot_df)