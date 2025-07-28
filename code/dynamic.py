import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFeatureIntegrationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicFeatureIntegrationModule, self).__init__()

        # 构建自我增强的全连接层
        self.self_enhancement_fc = nn.Linear(input_dim, hidden_dim)
        self.self_enhancement_out = nn.Linear(hidden_dim, input_dim)

        # 构建选择性过滤的网络层
        self.filter_fc = nn.Linear(input_dim, 1)

    def frame_level_self_enhancement(self, x):
        """进行帧级自我增强"""
        enhanced = F.relu(self.self_enhancement_fc(x))  # 激活函数
        enhanced = self.self_enhancement_out(enhanced)  # 输出增强特征
        return enhanced + x  # 增强特征与原始特征相加

    def selective_filtering(self, x):
        """进行选择性过滤"""
        # 使用全连接层为每个特征分配一个重要性评分
        scores = torch.sigmoid(self.filter_fc(x))  # 获取每个特征的选择性得分
        return x * scores  # 按照得分过滤特征

    def forward(self, text_features, audio_features, video_features):
        """
        输入：文本、音频、视频特征
        输出：融合后的增强特征
        """
        # 对每个模态的特征进行自我增强
        text_enhanced = self.frame_level_self_enhancement(text_features)
        audio_enhanced = self.frame_level_self_enhancement(audio_features)
        video_enhanced = self.frame_level_self_enhancement(video_features)

        # 对每个模态的增强特征进行选择性过滤
        text_filtered = self.selective_filtering(text_enhanced)
        audio_filtered = self.selective_filtering(audio_enhanced)
        video_filtered = self.selective_filtering(video_enhanced)

        # 融合三个模态的特征
        fused_features = torch.cat([text_filtered, audio_filtered, video_filtered], dim=-1)

        return fused_features
