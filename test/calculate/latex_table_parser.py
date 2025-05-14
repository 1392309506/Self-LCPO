import re


def parse_latex_table(filename):
    """解析 LaTeX 表格数据"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    current_model = None

    for line in lines:
        line = line.strip()
        # 匹配 \multirow 开头的模型行
        model_match = re.match(r'\\multirow\{.*\}\{.*\}\{(.*?)\}', line)
        if model_match:
            current_model = model_match.group(1)
            continue

        # 匹配数据行 (以 & 开头)
        if line.startswith('&'):
            # 移除 LaTeX 命令和特殊格式
            clean_line = re.sub(r'\\textbf\{|\}|\\text\{-\}|\\text\{.*?\}|\(.*?\)', '', line)
            clean_line = clean_line.replace('\\', '').strip()
            # 分割单元格
            cells = [cell.strip() for cell in clean_line.split('&')]
            # 移除空单元格和行尾的 \\
            cells = [cell for cell in cells if cell and not cell.endswith('\\')]
            # 添加模型名称作为第一列
            if current_model and cells:
                data.append([current_model] + cells)

    return data


def calculate_increase(data):
    """计算 Token 增长百分比"""
    # 找出所有包含 Token 的列 (从0开始，奇数列)
    token_cols = [i for i in range(2, len(data[0])) if i % 2 == 1]

    # 按模型分组
    models = {}
    for row in data:
        model = row[0]
        if model not in models:
            models[model] = []
        models[model].append(row)

    # 对每个模型计算增长
    for model, rows in models.items():
        if len(rows) < 6:  # 需要至少有6行数据
            continue

        # 第5-6行是基线 (索引4-5)
        # 第7-8行是升级后 (索引6-7)
        for i in range(2):
            baseline_row = rows[4 + i]
            upgraded_row = rows[6 + i]

            for col in token_cols:
                try:
                    # 提取原始Token值 (忽略百分比注释)
                    base_val = float(re.sub(r'\(.*', '', baseline_row[col]).strip())
                    upgraded_val = float(re.sub(r'\(.*', '', upgraded_row[col]).strip())

                    # 计算变化百分比
                    change = (upgraded_val - base_val) / base_val * 100

                    # 确定符号和格式
                    symbol = r'$\uparrow' if change >= 0 else r'$\downarrow'
                    formatted = f"({symbol} {abs(change):.1f}\\%$)"

                    # 添加到升级行
                    if '(' in upgraded_row[col]:
                        upgraded_row[col] = re.sub(r'\(.*', f' {formatted}', upgraded_row[col])
                    else:
                        upgraded_row[col] += f' {formatted}'
                except (ValueError, TypeError):
                    continue


def generate_latex_output(data, output_file):
    """生成 LaTeX 格式的输出"""
    with open(output_file, 'w') as f:
        current_model = None
        model_count = 0

        for row in data:
            if row[0] != current_model:
                current_model = row[0]
                model_count = 0
                f.write(f"\\multirow{{6}}{{*}}{{{current_model}}}\n")

            model_count += 1
            method = row[1]
            cells = row[2:]

            # 重建 LaTeX 行
            line = f"& {method.ljust(10)} "
            for i, cell in enumerate(cells):
                # 处理可能存在的数学模式
                if '$' in cell:
                    line += f"& {cell} "
                else:
                    line += f"& {cell} "

            line += r"\\"

            # 添加中间线
            if model_count == 6:
                line += "\n\\midrule"

            f.write(line + "\n")


if __name__ == "__main__":
    input_file = "latex_table.txt"
    output_file = "latex_table_processed.txt"

    # 1. 解析 LaTeX 表格
    table_data = parse_latex_table(input_file)
    print(table_data)

    # 2. 计算增长百分比
    calculate_increase(table_data)

    # 3. 生成新的 LaTeX 表格
    generate_latex_output(table_data, output_file)

    print(f"处理完成！结果已保存到 {output_file}")