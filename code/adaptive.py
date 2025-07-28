import torch
import torch.nn as nn
from transformers import BertModel

# 自适应层
class AdaptiveLayer(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))  # 缩放因子，初始化为1
        self.beta = nn.Parameter(torch.zeros(input_dim))  # 偏移因子，初始化为0

    def forward(self, x, mask=None):
        """
        对输入特征应用缩放和偏移操作
        :param x: 输入特征（Tensor）
        :param mask: 对输入进行掩码的二值张量 (1 - 有效，0 - 无效)
        :return: 调整后的特征
        """
        if mask is not None:
            x = x * mask  # 忽略缺失数据（乘以0）
        return self.gamma * x + self.beta

# 文本编码器
# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        # 修改为本地模型路径
        model_path = r'E:\多模态数据融合\bert-base-uncased'
        self.text_model = BertModel.from_pretrained(model_path)
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_fc(outputs.pooler_output)

# 音频编码器
class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.audio_fc = nn.Linear(256, 512)

    def forward(self, audio_features):
        return self.audio_fc(audio_features)

# 视频编码器
class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.video_fc = nn.Linear(512, 512)

    def forward(self, video_features):
        return self.video_fc(video_features)

# 缺失模态信息处理
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()

        # 模态编码器
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()

        # 自适应层
        self.text_adaptive_layer = AdaptiveLayer(512)
        self.audio_adaptive_layer = AdaptiveLayer(512)
        self.video_adaptive_layer = AdaptiveLayer(512)

        # 融合层
        self.fusion_fc = nn.Linear(512 * 3, 256)

        # 输出层
        self.classifier = nn.Linear(256, 2)

    def forward(self, text_inputs, audio_features, video_features, text_mask=None, audio_mask=None, video_mask=None):
        """
        :param text_inputs: 文本输入数据
        :param audio_features: 音频特征
        :param video_features: 视频特征
        :param text_mask: 文本掩码，1表示有效，0表示缺失
        :param audio_mask: 音频掩码，1表示有效，0表示缺失
        :param video_mask: 视频掩码，1表示有效，0表示缺失
        :return: 分类结果
        """
        # 文本特征
        text_features = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])
        text_features = self.text_adaptive_layer(text_features, mask=text_mask)

        # 音频特征
        audio_features = self.audio_encoder(audio_features)
        audio_features = self.audio_adaptive_layer(audio_features, mask=audio_mask)

        # 视频特征
        video_features = self.video_encoder(video_features)
        video_features = self.video_adaptive_layer(video_features, mask=video_mask)

        # 融合所有模态的特征（仅使用有效模态的特征）
        fused_features = torch.cat((text_features, audio_features, video_features), dim=-1)
        fused_features = self.fusion_fc(fused_features)

        # 分类输出
        output = self.classifier(fused_features)

        return output

# 示例输入和掩码
text_inputs = {'input_ids': torch.randint(0, 1000, (2, 10)), 'attention_mask': torch.ones(2, 10)}
audio_features = torch.randn(2, 256)
video_features = torch.randn(2, 512)

# 模态掩码，1表示该模态有效，0表示该模态缺失
text_mask = torch.tensor([1, 0], dtype=torch.float32).view(-1, 1)  # 第二个样本的文本缺失
audio_mask = torch.tensor([1, 1], dtype=torch.float32).view(-1, 1)  # 音频是有效的
video_mask = torch.tensor([1, 0], dtype=torch.float32).view(-1, 1)  # 第二个样本的视频缺失

# 模型实例化
model = MultiModalModel()

# 模型前向传播
output = model(text_inputs, audio_features, video_features, text_mask=text_mask, audio_mask=audio_mask, video_mask=video_mask)
print(output)
