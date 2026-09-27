import numpy as np

from myenv import UnoEnv
from myagent import SarsaLambdaAgent, RandomAgent

env = UnoEnv()

agent = SarsaLambdaAgent(
    env,
    gamma=0.99,
    alpha=0.05,
    eps=0.1,
    lam=0,
    trace="accumulating",
    total_epi=5000,
    init_val=1.0,
    seed=42
)

returns = agent.learn()

episode, terminated = agent.best_run()

print(episode)
print("Terminated:", terminated)

eligibility = np.zeros_like(agent.q)

eligibility[5, 1] += 1

eligibility *= agent.gamma * agent.lam

print(eligibility)