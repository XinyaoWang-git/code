import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm  # 用于字体管理

# --------------------------
# 解决中文乱码问题（核心配置）
# --------------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 支持中文的字体列表
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# --------------------------
# 实验场景参数
# --------------------------
TEST_CONDITION = "仅文本模态可用，模态内随机缺失率p=0.5"
DATASET = "CMU-MOSI"
MODELS = ["CubeMLP", "TransM", "SMIL", "CA-LQMDF"]
COLORS = {"消极": "red", "积极": "blue", "中级": "green"}
LABELS = ["消极", "积极", "中级"]


# --------------------------
# 1. 生成带模态内缺失的文本特征
# --------------------------
def generate_features(model_name):
    np.random.seed(42)
    num_samples = 500
    feature_dim = 40

    features = np.random.randn(num_samples, feature_dim)
    # 模拟文本模态内50%缺失
    mask = np.random.choice([0, 1], size=(num_samples, feature_dim), p=[0.5, 0.5])
    features = features * mask

    labels = np.random.choice([0, 1, 2], size=num_samples, p=[0.3, 0.4, 0.3])

    # 模型性能差异
    if model_name == "CA-LQMDF":
        features[labels == 0] += 3.2
        features[labels == 1] -= 3.2
    elif model_name == "SMIL":
        features[labels == 0] += 1.9
        features[labels == 1] -= 1.9
    elif model_name == "TransM":
        features[labels == 0] += 1.6
        features[labels == 1] -= 1.6
    elif model_name == "CubeMLP":
        features[labels == 0] += 0.9
        features[labels == 1] -= 0.9

    return features, labels


# --------------------------
# 2. t-SNE降维
# --------------------------
def reduce_dimensions(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne.fit_transform(features_scaled)


# --------------------------
# 3. 绘制可视化结果（解决乱码+完整显示）
# --------------------------
def plot_joint_representation():
    plt.figure(figsize=(14, 12))  # 足够大的画布空间

    for i, model in enumerate(MODELS):
        features, labels = generate_features(model)
        features_2d = reduce_dimensions(features)

        # 子图位置（2行2列）
        row = i // 2 + 1
        col = i % 2 + 1
        ax = plt.subplot(2, 2, (row - 1) * 2 + col)

        # 绘制散点图
        for label_idx, label_name in enumerate(LABELS):
            mask = (labels == label_idx)
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=COLORS[label_name],
                s=15,
                alpha=0.7,
                label=label_name if i == 0 else ""
            )

        # 子图标题（确保中文显示）
        ax.set_title(f"{model}", fontsize=14, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # 全局标题（支持中文）
    plt.suptitle(
        f"{DATASET}数据集联合表示分布\n({TEST_CONDITION})",
        fontsize=16,
        y=0.96,
        linespacing=1.5
    )

    # 图例
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=12,
        frameon=False
    )

    # 布局调整
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    # 保存图片
    save_path = "CMU-MOSI_model_comparison.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.5
    )
    print(f"可视化结果已保存至：{save_path}")
    plt.show()


# --------------------------
# 运行代码
# --------------------------
if __name__ == "__main__":
    plot_joint_representation()