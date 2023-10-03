import pandas as pd

# 创建包含数据的字典
data = {
    'prob': [0.000000, 0.000500, 0.001000, 0.001500, 0.002000, 0.002500, 0.003000, 0.003500, 0.004000, 0.004500,
             0.005000, 0.005500, 0.006000, 0.006500, 0.007000, 0.007500, 0.008000, 0.008500, 0.009000, 0.009500,
             0.010000, 0.010500, 0.011000, 0.011500, 0.012000, 0.012500, 0.013000, 0.013500, 0.014000],
    'accuracy': [0.980500, 0.974400, 0.969100, 0.960400, 0.953200, 0.938100, 0.930100, 0.913800, 0.905600, 0.889600,
                 0.878600, 0.859600, 0.851400, 0.828800, 0.814400, 0.798400, 0.786300, 0.770800, 0.750500, 0.734700,
                 0.724300, 0.707000, 0.694100, 0.677600, 0.664000, 0.650100, 0.637400, 0.620300, 0.614000]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 将DataFrame转置（行变成列，列变成行）
df = df.T

# 将DataFrame写入Excel文件
excel_file = "Nofix50accuracy_data.xlsx"
df.to_excel(excel_file, index=False, header=False)  # 不写入行和列的索引，不写入列名

print(f"Data has been written to {excel_file}")