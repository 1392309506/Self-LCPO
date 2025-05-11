# Plotting the ACC vs Token Settings for StrategyQA and BBH-Navigate
import matplotlib.pyplot as plt

suggested_token = [100, 500, 900, 1300, 1700, 2100, 2500, 2900]
# BBH-Navigate data
actual_token_bbh = [1750.0, 3554.02, 5487.52, 7639.44, 9572.55, 11249.59, 13052.21, 15075.27]
acc_scores_bbh = [0.8283, 0.7800, 0.8400, 0.8100, 0.7900, 0.8700, 0.7800, 0.8367]
# StrategyQA data
# suggested_token_strategyqa = [100, 200, 300, 400, 500, 600, 700, 800]
# actual_token_strategyqa = [586.06, 632.72, 568.54, 1197.22, 1171.1, 610.3, 575.0, 610.3]
# acc_scores_strategyqa = [0.7000, 0.6800, 0.7900, 0.6900, 0.7500, 0.7600, 0.7400, 0.7600]
actual_token_strategyqa = [588.31, 1134.45, 1720.84, 2285.29, 2812.04, 3421.54, 577.06, 1179.39]
acc_scores_strategyqa = [0.4600, 0.4400, 0.4400, 0.4800, 0.4300, 0.5400, 0.4300, 0.4200]


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plotting ACC vs Suggested Token on the first subplot
axs[0].plot(suggested_token, acc_scores_bbh, marker='o', linestyle='-', color='b', label='BBH-Navigate')
axs[0].plot(suggested_token, acc_scores_strategyqa, marker='o', linestyle='-', color='g', label='StrategyQA')
axs[0].set_title("(a)ACC Score vs Suggested Token", fontsize=16, fontweight='bold')
axs[0].set_xlabel("Suggested Token", fontsize=14, fontweight='bold')
axs[0].set_ylabel("ACC Score", fontsize=14, fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].legend(loc='upper left', fontsize=12)
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].set_xticks(suggested_token)
axs[0].set_ylim(0.3, 0.98)  # Adjust this range to make the lines closer


# Plotting Actual token vs Suggested Token on the second subplot
axs[1].plot(suggested_token, actual_token_bbh, marker='o', linestyle='-', color='b', label='BBH-Navigate')
axs[1].plot(suggested_token, actual_token_strategyqa, marker='o', linestyle='-', color='g', label='StrategyQA')
axs[1].set_title("(b)Actual Token vs Suggested Token", fontsize=16, fontweight='bold')
axs[1].set_xlabel("Suggested Token", fontsize=14, fontweight='bold')
axs[1].set_ylabel("Actual Token", fontsize=14, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].legend(loc='upper left', fontsize=12)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xticks(suggested_token)

# Show the plots with improved layout
plt.tight_layout()
plt.show()