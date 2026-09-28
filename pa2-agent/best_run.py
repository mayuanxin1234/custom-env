import sys
import os

for path in ['/opt/homebrew/lib/python3.13/site-packages', '/opt/homebrew/lib/python3.11/site-packages']:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

from myenv import UnoEnv
from myagent import SarsaLambdaAgent

def run_best_run():
    print("=== Training SARSA(lambda=0.9) Agent for Best Run Evaluation ===")
    env = UnoEnv(render_mode="ansi")
    
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

    # Train agent first
    agent.learn()
    print("✓ Training completed (5000 episodes). Executing greedy best run...")

    # Execute greedy policy (exploration=False)
    episode, terminated = agent.best_run(max_steps=300)
    total_return = agent.calc_return(episode, discounted=False)
    discounted_return = agent.calc_return(episode, discounted=True)

    print("\n" + "="*50)
    print("          GREEDY EPISODE TRAJECTORY")
    print("="*50)
    for step_idx, (state, action, reward) in enumerate(episode):
        action_str = "Play (0)" if action == 0 else "Draw (1)"
        print(f"Step {step_idx+1:2d} | State: {state:3d} | Action: {action_str:<8} | Reward: {reward:+5.1f}")

    print("="*50)
    print(f"Total Steps: {len(episode)}")
    print(f"Undiscounted Return: {total_return:.2f}")
    print(f"Discounted Return (gamma=0.99): {discounted_return:.4f}")
    print(f"Clean Termination: {terminated}")
    print("="*50)

if __name__ == "__main__":
    run_best_run()