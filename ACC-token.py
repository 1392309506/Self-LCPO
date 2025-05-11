# Data for BBH-Navigate
from matplotlib import pyplot as plt

avg_token_bbh = [1750.02, 3554.02, 5487.52, 7639.44, 9572.55, 11249.59, 13052.21, 15075.27]
acc_score_bbh = [0.8283, 0.7800, 0.8400, 0.8100, 0.7900, 0.8700, 0.7800, 0.8367]

#Re-checking StrategyQA data and making sure all points are included
avg_token_strategyqa_full = [586.06, 632.72, 568.54, 1197.22, 1171.1, 610.3, 575.0, 610.3, 1149.92, 1809.81]
acc_score_strategyqa_full = [0.7000, 0.6800, 0.7900, 0.6900, 0.7500, 0.7600, 0.7400, 0.7600, 0.7600, 0.7300]

# Sorting the StrategyQA data correctly
sorted_data_strategyqa_full = sorted(zip(avg_token_strategyqa_full, acc_score_strategyqa_full))
sorted_avg_token_strategyqa_full, sorted_acc_score_strategyqa_full = zip(*sorted_data_strategyqa_full)

# Removing the specific data point from StrategyQA
filtered_avg_token_strategyqa = [token for token, acc in zip(sorted_avg_token_strategyqa_full, sorted_acc_score_strategyqa_full) if token != 575.0]
filtered_acc_score_strategyqa = [acc for token, acc in zip(sorted_avg_token_strategyqa_full, sorted_acc_score_strategyqa_full) if token != 575.0]

# Improved plotting style
plt.figure(figsize=(10, 6))

# BBH-Navigate with smoother lines and distinct markers
plt.plot(avg_token_bbh, acc_score_bbh, marker='o', linestyle='-', color='b', markersize=8, linewidth=2, label='BBH-Navigate')

# StrategyQA with smoother lines and larger markers
plt.plot(filtered_avg_token_strategyqa, filtered_acc_score_strategyqa, marker='o', linestyle='-', color='g', markersize=8, linewidth=2, label='StrategyQA')

# Title and labels with enhanced styling
plt.title("ACC Score vs Actual Token", fontsize=16, fontweight='bold')
plt.xlabel("Actual Token", fontsize=14, fontweight='bold')
plt.ylabel("ACC Score", fontsize=14, fontweight='bold')

# Adding gridlines with lighter style
plt.grid(True, linestyle='--', alpha=0.7)

# Adding legend with improved readability
plt.legend(loc='upper left', fontsize=12)

# Show plot with improved layout
plt.tight_layout()
plt.show()
