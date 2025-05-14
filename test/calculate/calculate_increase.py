import re
from collections import defaultdict


def parse_latex_table(content):
    """解析LaTeX表格内容"""
    models = defaultdict(list)
    current_model = None

    for line in content.split('\n'):
        line = line.strip()
        # 匹配模型行
        model_match = re.match(r'\\multirow\{.*\}\{.*\}\{(.*?)\}', line)
        if model_match:
            current_model = model_match.group(1)
            continue

        # 匹配数据行
        if line.startswith('&'):
            # 保存原始行
            if current_model:
                models[current_model].append(line)

    return models


def calculate_token_changes(models):
    """计算Token变化百分比"""
    processed_models = defaultdict(list)

    for model, rows in models.items():
        # 确保有足够的数据行
        if len(rows) < 6:
            print(f"警告: 模型 {model} 数据不完整，跳过计算")
            processed_models[model] = rows
            continue

        # 处理每组6行数据
        for i in range(2):  # 每组比较2对行
            baseline_idx = 3 + i  # 第4或第5行作为基线
            upgraded_idx = 5 + i  # 第6或第7行作为升级

            if upgraded_idx >= len(rows):
                continue

            baseline_line = rows[baseline_idx]
            upgraded_line = rows[upgraded_idx]

            # 提取Token值
            baseline_tokens = extract_tokens(baseline_line)
            upgraded_tokens = extract_tokens(upgraded_line)

            # 计算变化并修改行
            modified_line = add_percentage_changes(upgraded_line, baseline_tokens, upgraded_tokens)
            rows[upgraded_idx] = modified_line

        processed_models[model] = rows

    return processed_models


def extract_tokens(line):
    """从行中提取Token值"""
    cells = [cell.strip() for cell in line.split('&')]
    # Token位于第4,6,8,10,12位置(从1开始计数)
    token_positions = [3, 5, 7, 9, 11]
    tokens = []

    for pos in token_positions:
        if pos < len(cells):
            cell = cells[pos]
            # 移除已有百分比和格式
            clean_cell = re.sub(r'\(.*?\)', '', cell)
            clean_cell = re.sub(r'\\textbf\{|\}', '', clean_cell)
            clean_cell = re.sub(r'\\text\{-\}', '-', clean_cell)
            tokens.append(clean_cell.strip())
        else:
            tokens.append(None)

    return tokens


def add_percentage_changes(line, baseline_tokens, upgraded_tokens):
    """在行中添加百分比变化"""
    cells = [cell.strip() for cell in line.split('&')]
    token_positions = [3, 5, 7, 9, 11]  # Token列位置

    for i, pos in enumerate(token_positions):
        if pos >= len(cells):
            continue

        base = baseline_tokens[i]
        up = upgraded_tokens[i]

        # 跳过无效数据
        if not base or not up or base == '-' or up == '-':
            continue

        try:
            base_val = float(base)
            up_val = float(up)
            change = (up_val - base_val) / base_val * 100

            # 确定符号和格式
            symbol = r'$\uparrow' if change >= 0 else r'$\downarrow'
            change_str = f"({symbol} {abs(change):.1f}\\%)"

            # 添加到单元格
            cell = cells[pos]
            if '(' not in cell:  # 如果没有已有注释
                cells[pos] = f" {cell}{change_str} "
        except (ValueError, TypeError):
            continue

    return '&'.join(cells)


def generate_output(models):
    """生成最终输出"""
    output = []
    for model, rows in models.items():
        output.append(f"\\multirow{{6}}{{*}}{{{model}}}")
        output.extend(rows)
        output.append("\\midrule")
    return '\n'.join(output)


def process_latex_table(input_file, output_file):
    """处理LaTeX表格文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    models = parse_latex_table(content)
    models = calculate_token_changes(models)
    output = generate_output(models)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)


if __name__ == "__main__":
    input_file = "latex_table.txt"  # 替换为您的输入文件
    output_file = "output_table.tex"  # 输出文件
    process_latex_table(input_file, output_file)