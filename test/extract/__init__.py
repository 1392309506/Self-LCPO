# 输入和输出文件
input_file = "input.txt"
output_file = "output.txt"

# 1. 读取原始数据
with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]

# 2. 提取列名（boolq, str, gpqa, bbh, wsc）
columns = lines[0].split("\t")

# 3. 初始化数据存储字典
data = {col: [] for col in columns}

# 4. 解析数据行（跳过表头行）
for line in lines[1:]:
    parts = line.split("\t")
    for col, value in zip(columns, parts):
        # 提取数值部分（移除 "Mean:" 等前缀）
        num_value = value.split(":")[-1].strip()
        data[col].append(num_value)  # 保持字符串格式，不转 float

# 5. 定义行索引（统计指标名称）
index = [
    "Count", "Mean", "Std Dev", "Min", "Max",
    "Zscore Min", "Zscore Max", "Skewness", "Kurtosis"
]

# 6. 写入到 output.txt
with open(output_file, "w", encoding="utf-8") as f:
    # 写入列名（用制表符分隔）
    f.write("\t".join(columns) + "\n")

    # 逐行写入数据
    for i, metric in enumerate(index):
        row = [metric]  # 行首是指标名称
        for col in columns:
            row.append(data[col][i])  # 添加对应数据
        f.write("\t".join(row) + "\n")

print(f"✅ 数据已成功导出到 {output_file}")