import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel表格
excel_file = "conv:accuracy_data.xlsx"  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file)

# 提取横坐标和折线名称
x_label = df.iloc[0, 1:].tolist()  # 第一行，从第二列开始到最后一列
x_name = df.iloc[0, 0]  # 第一行第一列

# 提取折线数据和折线名称
lines = []
for i in range(1, len(df)):
    line_data = df.iloc[i, 1:].tolist()  # 从第二列到最后一列
    line_name = df.iloc[i, 0]  # 第一列
    lines.append((line_name, line_data))

# 创建折线图
plt.figure(figsize=(10, 6))  # 设置图形大小

for line_name, line_data in lines:
    plt.plot(x_label, line_data, label=line_name)

plt.xlabel(x_name)  # 设置横坐标名称
plt.ylabel("ACC")  # 设置纵坐标名称
plt.title("Accuracy vs. " + x_name)  # 设置图表标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线

plt.show()
