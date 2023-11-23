import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib as mpl
from termcolor import colored

# load data
df = pd.read_csv("./entity_res/results_papers_cs_8k.csv")

# 需要排除的值集合
excluded_names = [
    "Not mentioned in the paper",
    "Not mentioned",
    "Not mentioned in the provided materials",
    "No specific dataset mentioned",
    "Not mentioned in the provided information",
    "Synthetic Data",
    "Synthetic dataset",
    "Unknown",
    "Not provided",
]

# 排除指定的数据集名称
df = df[~df["Dataset Name"].isin(excluded_names)]


# 创建一个函数，将数据集名称的单词排序并连接，以生成一个标识符
def create_Identifier(name):
    if isinstance(name, str):
        cleaned_name = re.sub(r"[^A-Za-z0-9]+", "", name)
        return "".join(sorted(cleaned_name.lower().split()))
    return None  # 对于非字符串类型，返回None


# 对每个数据集名称应用此函数
df["Identifier"] = df["Dataset Name"].apply(create_Identifier)

print(df.shape)
# 删除那些没有标识符的行（即，原始数据集名称不是字符串类型的行）
print(df[df["Identifier"].isna()].head)
df = df[df["Identifier"].notna()]
print(df.shape)
# 写入新的CSV文件
# df.to_csv('./entity_res/paper_dataset_network.csv', index=False)
# exit(0)

# 统计每个标识符出现的次数
Identifier_counts = df["Identifier"].value_counts()

# # 过滤掉计数少于或等于2的条目
# Identifier_counts = Identifier_counts[Identifier_counts > 1]

# 计算每个原始名称的出现次数
name_counts = df["Dataset Name"].value_counts()

# 找出每个标识符对应的原始名称（出现次数最多的那个）
original_names = {}
for Identifier, name in zip(df["Identifier"], df["Dataset Name"]):
    if Identifier not in original_names:
        original_names[Identifier] = name
    else:
        # 获取此标识符已存储名称的计数
        stored_name = original_names[Identifier]
        stored_count = name_counts[stored_name]

        # 获取当前名称的计数
        current_count = name_counts[name]

        # 如果当前名称的计数更高，更新字典
        if current_count > stored_count:
            original_names[Identifier] = name

# 创建一个新的计数器，其中键是原始名称，值是出现次数
aggregated_counts = {}
for Identifier, name in original_names.items():
    if Identifier in Identifier_counts:
        aggregated_counts[name] = Identifier_counts[Identifier]

# 对 aggregated_counts 按值（出现次数）降序排序，并获取前n个
top_datasets = sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# 打印结果
for name, count in top_datasets:
    print(f"{name}: {count}")

print("*" * 150)

# 对 aggregated_counts 按值（出现次数）降序排序，并获取前10个
# top_10_datasets = sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# 计算总出现次数
total_counts = sum(aggregated_counts.values())

# 提取数据集名称和计数，以便绘图
datasets, counts = zip(*top_datasets)

# 计算占比
proportions = [(count / total_counts) * 100 for count in counts]


# 绘图
# # 创建颜色列表
# colors = plt.cm.viridis(np.linspace(0.35, 1.0, len(datasets)))  # 这里使用 'viridis' 颜色图，您可以根据喜好更改

# # 设置全局字体样式和大小
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['font.size'] = 10  # 可以选择8-10之间的值，这里我们设为10

# # 创建柱状图
# plt.figure(figsize=(11.5, 4.5))
# bars = plt.barh(datasets, counts, color=colors)  # 使用颜色列表

# # 在每个条形上添加占比
# for bar, proportion in zip(bars, proportions):
#     plt.text(
#       bar.get_width(),  # 获取条形的宽度，即数据集的计数
#       bar.get_y() + bar.get_height() / 2,  # 获取条形的垂直位置（居中）
#       ' {:.2f}%'.format(proportion),  # 格式化为百分比，保留两位小数
#       va='center',  # 垂直对齐方式
#       ha='left',  # 水平对齐方式
#       fontsize=10,  # 由于全局字体大小已设为10，此处的设定其实是可选的
#     )

# plt.grid(True, linestyle='--', linewidth=0.5, axis='x', alpha=0.7)  # 添加网格线
# plt.xlabel('Counts')
# plt.title(f'Top {len(top_datasets)} Most Frequent Datasets with Proportions')
# plt.gca().invert_yaxis()  # 反转Y轴，使得最高频的数据集在顶部
# plt.tight_layout()  # 调整布局以确保图表完整显示
# # plt.show()
# plt.savefig("dataset_distribution.pdf", format='pdf', bbox_inches='tight')
