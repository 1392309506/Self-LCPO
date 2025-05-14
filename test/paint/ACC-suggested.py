# Plotting the ACC vs Token Settings for StrategyQA and BBH-Navigate
import matplotlib.pyplot as plt

suggested_token = [100, 500, 900, 1300, 1700, 2100, 2500, 2900]
# BBH-Navigate data
actual_token_bbh = [1750.02, 1804, 1933.5, 2152.02, 1933.11, 1677.04, 1802.62, 2023.05]
acc_scores_bbh = [0.8283, 0.7800, 0.8400, 0.8100, 0.7900, 0.8700, 0.7800, 0.8367]
# StrategyQA data
actual_token_strategyqa = [588.31, 546.14, 586.39,564.45, 526.75, 609.5, 577.06, 602.33]
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