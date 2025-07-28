import torch
import torch.nn as nn

class CrossModalInteractionLayer(nn.Module):
    def __init__(self, input_dim, interaction_type='coarse', num_heads=2):
        """
        一个支持粗粒度、中粒度和细粒度交互的交叉模态交互层
        :param input_dim: 输入的特征维度
        :param interaction_type: 'coarse'/'medium'/'fine'，交互粒度类型
        :param num_heads: 注意力机制的头数
        """
        super(CrossModalInteractionLayer, self).__init__()
        self.interaction_type = interaction_type

        if interaction_type == 'coarse':
            # 粗粒度交互，直接将三个模态拼接
            self.fc = nn.Linear(input_dim * 3, input_dim)

        elif interaction_type == 'medium':
            # 中粒度交互，采用注意力机制进行模态间交互
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
            self.fc = nn.Linear(input_dim, input_dim)

        elif interaction_type == 'fine':
            # 细粒度交互，先通过全连接层再进行处理
            self.fc1 = nn.Linear(input_dim, input_dim * 2)
            self.fc2 = nn.Linear(input_dim * 2, input_dim)

    def forward(self, text, audio, video):
        if self.interaction_type == 'coarse':
            combined = torch.cat((text, audio, video), dim=-1)
            return torch.relu(self.fc(combined))

        elif self.interaction_type == 'medium':
            combined = torch.stack((text, audio, video), dim=0)
            attention_output, _ = self.attention(combined, combined, combined)
            output = attention_output.mean(dim=0)  # 聚合结果
            return torch.relu(self.fc(output))

        elif self.interaction_type == 'fine':
            combined = torch.cat((text, audio, video), dim=-1)
            intermediate = torch.relu(self.fc1(combined))
            return self.fc2(intermediate)


class MultiLayerCrossModalInteraction(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        """
        包含多层交叉模态交互层的模块，处理粗粒度、中粒度、细粒度交互
        :param input_dim: 输入的特征维度
        :param num_heads: 注意力头数
        """
        super(MultiLayerCrossModalInteraction, self).__init__()
        self.coarse_layer = CrossModalInteractionLayer(input_dim, interaction_type='coarse', num_heads=num_heads)
        self.medium_layer = CrossModalInteractionLayer(input_dim, interaction_type='medium', num_heads=num_heads)
        self.fine_layer = CrossModalInteractionLayer(input_dim, interaction_type='fine', num_heads=num_heads)
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, text, audio, video):
        # 经过粗粒度交互
        coarse_output = self.coarse_layer(text, audio, video)

        # 经过中粒度交互
        medium_output = self.medium_layer(coarse_output, coarse_output, coarse_output)

        # 经过细粒度交互
        fine_output = self.fine_layer(medium_output, medium_output, medium_output)

        # 最终分类输出
        output = self.classifier(fine_output)
        return output
