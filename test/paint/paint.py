import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')  # 适配服务器或无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

from utils.logger_utils import LoggerUtil
logger = LoggerUtil.get_logger("Paint")
def main():
    args = parse_args()
    logger.info(args)
    # 读取数据文件路径
    file_path = args.file_path

    # 图像与数据输出路径
    output_dir = 'figures'
    if file_path.startswith('results/exp_prompt/'):
        relative_subpath = file_path[len('results/exp_prompt/'):]  # '20250503/...'
    relative_dir = os.path.dirname(relative_subpath)  # '20250503/20250503_ds_boolq'
    # 构造输出路径
    output_dir = os.path.join('figures', relative_dir)
    os.makedirs(output_dir, exist_ok=True)  # 自动创建多级目录

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取token_consumed数据并确保是整数
    tokens = []
    for entry in data:
        if 'token_consumed' in entry:
            try:
                value = int(entry['token_consumed'])
                tokens.append(value)
            except ValueError:
                print(f"跳过非法 token 值: {entry['token_consumed']}")

    # 计算统计量
    tokens_np = np.array(tokens)
    stats_summary = {
        'Count': len(tokens),
        'Mean': np.mean(tokens_np),
        'Std Dev': np.std(tokens_np),
        'Min': np.min(tokens_np),
        'Max': np.max(tokens_np),
        'Skewness': stats.skew(tokens_np),
        'Kurtosis': stats.kurtosis(tokens_np),
    }

    # -------- 图像 1: 直方图 + KDE --------
    plt.style.use('ggplot')

    plt.figure(figsize=(8, 5))
    sns.histplot(tokens, kde=True, bins=30, color='salmon')
    plt.title('Token Consumption Distribution')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_distribution.png'))
    plt.close()

    # -------- 图像 2: Q-Q Plot --------
    plt.figure(figsize=(6, 5))
    stats.probplot(tokens, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Check Normality)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_qqplot.png'))
    plt.close()

    # -------- 归一化 --------
    tokens_z = (tokens_np - stats_summary['Mean']) / stats_summary['Std Dev']

    # -------- 计算标准化后的统计量（Z-score）--------

    # 统一整理统计信息，调整顺序
    stats_all = {
        'Count': len(tokens),
        'Mean': np.mean(tokens_np),
        'Std Dev': np.std(tokens_np),
        'Min': np.min(tokens_np),
        'Max': np.max(tokens_np),
        'Zscore Min': np.min(tokens_z),
        'Zscore Max': np.max(tokens_z),
        'Skewness': stats.skew(tokens_np),
        'Kurtosis': stats.kurtosis(tokens_np),
    }

    # 写入统计信息（覆盖写入）
    with open(os.path.join(output_dir, 'token_stats.txt'), 'w') as f:
        for k, v in stats_all.items():
            f.write(f"{k}: {v:.4f}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='大模型思考长度实验')
    parser.add_argument('--file_path', type=str, default="results/exp_prompt/20250505/20250505_ds_gpqa/results_data.json", help="目录")
    return parser.parse_args()

if __name__ == "__main__":
    main()
