import numpy as np
import matplotlib.pyplot as plt

from myenv import UnoEnv

from myagent import SarsaLambdaAgent

lambdas = [0, 0.3, 0.6, 0.9, 1.0]
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


window = 100

plt.figure(figsize=(10, 6))

for lam in lambdas:

    data = np.array(all_returns[lam])

    mean_returns = data.mean(axis=0)

    std_returns = data.std(axis=0)

    kernel = np.ones(window) / window

    mean_smooth = np.convolve(
        mean_returns,
        kernel,
        mode="valid"
    )

    std_smooth = np.convolve(
        std_returns,
        kernel,
        mode="valid"
    )

    episodes = np.arange(window, len(mean_returns) + 1)

    plt.plot(
        episodes,
        mean_smooth,
        label=f"λ={lam}"
    )

    plt.fill_between(
        episodes,
        mean_smooth - std_smooth,
        mean_smooth + std_smooth,
        alpha=0.2
    )

plt.xlabel("Episode")
plt.ylabel("Mean Return per episode")
plt.title("SARSA(λ) Learning Curves")
plt.legend()
plt.grid(True)
plt.show()