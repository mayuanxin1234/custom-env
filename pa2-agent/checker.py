import numpy as np
import gymnasium as gym

from myenv import UnoEnv
from myagent import SarsaLambdaAgent, RandomAgent

def run_checker():
    print("==================================================")
    print("   CHECKER: SARSA(lambda) vs RandomAgent Baseline")
    print("==================================================")

    seeds = [0, 1, 2, 3, 4]
    
    # 1. Random Agent Performance Baseline
    print("\n--- Evaluating RandomAgent Baseline across 5 seeds ---")
    random_returns = []
    for s in seeds:
        env = UnoEnv()
        rand_agent = RandomAgent(env, total_epi=1000, seed=s)
        ret = rand_agent.learn()
        random_returns.append(np.mean(ret))
        print(f"  Seed {s}: Mean return = {np.mean(ret):.4f}")

    avg_random = np.mean(random_returns)
    print(f">> RandomAgent Average Mean Return across seeds: {avg_random:.4f}")

    # 2. SARSA(lambda) Agent Training & Evaluation
    print("\n--- Training SarsaLambdaAgent (lambda=0.9, 5000 episodes) ---")
    env = UnoEnv()
    agent = SarsaLambdaAgent(
        env,
        gamma=0.99,
        alpha=0.05,
        eps=0.1,
        lam=0.9,
        trace="accumulating",
        total_epi=5000,
        init_val=1.0,
        seed=42
    )

    sarsa_returns = agent.learn()
    final_100_avg = np.mean(sarsa_returns[-100:])
    print(f">> SARSA(lambda=0.9) Final 100-episode Mean Return: {final_100_avg:.4f}")

    # 3. Greedy Best Run Execution
    print("\n--- Executing Best Run (Greedy Policy) ---")
    episode, terminated = agent.best_run(max_steps=300)
    greedy_return = agent.calc_return(episode, discounted=False)
    print(f"Greedy Episode Steps: {len(episode)}")
    print(f"Greedy Episode Return: {greedy_return}")
    print(f"Episode Terminated Cleanly: {terminated}")

    print("\n==================================================")
    print(" SUMMARY COMPARISON")
    print("==================================================")
    print(f"Random Agent Mean Return: {avg_random:.4f}")
    print(f"SARSA(lambda=0.9) Mean Final Return: {final_100_avg:.4f}")
    print(f"Performance Gain over Random: {final_100_avg - avg_random:+.4f}")

if __name__ == "__main__":
    run_checker()