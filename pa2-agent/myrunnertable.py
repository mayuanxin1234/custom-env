import sys
import os

for path in ['/opt/homebrew/lib/python3.13/site-packages', '/opt/homebrew/lib/python3.11/site-packages']:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

import numpy as np
import pandas as pd

from myenv import UnoEnv
from myagent import SarsaLambdaAgent

lambdas = [0.0, 0.3, 0.6, 0.9, 1.0]
seeds = [0, 1, 2, 3, 4]

all_returns = {}

for lam in lambdas:
    all_returns[lam] = []
    for seed in seeds:
        env = UnoEnv()
        agent = SarsaLambdaAgent(
            env,
            gamma=0.99,
            alpha=0.05,
            eps=0.1,
            lam=lam,
            trace="accumulating",
            total_epi=5000,
            init_val=1.0,
            seed=seed
        )
        returns = agent.learn()
        all_returns[lam].append(returns)

target_return = 1.0
window = 100

results = []

for lam in lambdas:
    data = np.array(all_returns[lam])
    first_reach = []

    for seed_returns in data:
        moving_avg = np.convolve(
            seed_returns,
            np.ones(window) / window,
            mode="valid"
        )
        reached = np.where(moving_avg >= target_return)[0]
        if len(reached) > 0:
            first_reach.append(reached[0] + window)
        else:
            first_reach.append(np.nan)

    mean_episodes = np.nanmean(first_reach)
    mean_final = data[:, -100:].mean()

    results.append({
        "λ": lam,
        "Episodes to target": mean_episodes,
        "Mean final return": mean_final
    })

results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("       SARSA(λ) EXPERIMENTAL RESULTS TABLE")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)