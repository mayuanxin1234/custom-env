import sys
import os

# Add site-packages path if needed
for path in ['/opt/homebrew/lib/python3.13/site-packages', '/opt/homebrew/lib/python3.11/site-packages']:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt

from myenv import UnoEnv
from myagent import SarsaLambdaAgent, RandomAgent

lambdas = [0.0, 0.3, 0.6, 0.9, 1.0]
seeds = [0, 1, 2, 3, 4]
total_epi = 5000
window = 100
target_return = 1.0

all_returns = {lam: [] for lam in lambdas}

print(f"Running SARSA(lambda) experiments for lambdas={lambdas} across seeds={seeds}...")

for lam in lambdas:
    print(f"  Training lambda = {lam}...")
    for seed in seeds:
        env = UnoEnv()
        agent = SarsaLambdaAgent(
            env,
            gamma=0.99,
            alpha=0.05,
            eps=0.1,
            lam=lam,
            trace="accumulating",
            total_epi=total_epi,
            init_val=1.0,
            seed=seed
        )
        returns = agent.learn()
        all_returns[lam].append(returns)

# Evaluate RandomAgent Baseline
random_all = []
for seed in seeds:
    env = UnoEnv()
    rand_agent = RandomAgent(env, total_epi=total_epi, seed=seed)
    r = rand_agent.learn()
    random_all.append(r)

random_mean = np.mean(random_all)

# Plotting Learning Curves
plt.figure(figsize=(10, 6))

colors = {0.0: "#e41a1c", 0.3: "#377eb8", 0.6: "#4daf4a", 0.9: "#984ea3", 1.0: "#ff7f00"}

for lam in lambdas:
    data = np.array(all_returns[lam])
    mean_returns = data.mean(axis=0)
    std_returns = data.std(axis=0)

    kernel = np.ones(window) / window
    mean_smooth = np.convolve(mean_returns, kernel, mode="valid")
    std_smooth = np.convolve(std_returns, kernel, mode="valid")

    episodes = np.arange(window, total_epi + 1)

    plt.plot(episodes, mean_smooth, label=f"λ = {lam}", color=colors[lam], linewidth=2.0)
    plt.fill_between(
        episodes,
        mean_smooth - std_smooth,
        mean_smooth + std_smooth,
        color=colors[lam],
        alpha=0.12
    )

plt.axhline(y=random_mean, color="black", linestyle="--", linewidth=1.5, label=f"Random Baseline ({random_mean:.2f})")
plt.xlabel("Training Episode", fontsize=12)
plt.ylabel(f"Mean Return per Episode (Window = {window})", fontsize=12)
plt.title("SARSA(λ) Learning Curves on Custom UNO Environment (cs272/Uno-v0)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "learning_curves.png")
plt.savefig(output_path, dpi=300)
print(f"✓ Saved learning curves figure to {output_path}")

# Print numerical results table
print("\n" + "="*65)
print(f"{'λ':<6} | {'Episodes to Target (≥1.0)':<25} | {'Final 100-Epi Mean Return':<25}")
print("="*65)

for lam in lambdas:
    data = np.array(all_returns[lam])
    first_reach = []

    for seed_returns in data:
        moving_avg = np.convolve(seed_returns, np.ones(window) / window, mode="valid")
        reached = np.where(moving_avg >= target_return)[0]
        if len(reached) > 0:
            first_reach.append(reached[0] + window)
        else:
            first_reach.append(np.nan)

    mean_episodes = np.nanmean(first_reach)
    mean_final = data[:, -100:].mean()

    ep_str = f"{mean_episodes:.1f}" if not np.isnan(mean_episodes) else "N/A"
    print(f"{lam:<6.1f} | {ep_str:<25} | {mean_final:<25.4f}")

print("="*65)