from gymnasium.utils.env_checker import check_env
from uno import UnoEnv
import gymnasium as gym

# Register the environment so we can create it with gym.make()
gym.register(
    id="test/Uno-v0",
    entry_point=UnoEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)

env = gym.make("test/Uno-v0", render_mode="human")
# This will catch many common issues
try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

obs, info = env.reset(seed=42)  # Use seed for reproducible testing

print(f"observation: {obs}")

# Test each action type
actions = [0, 1]  # 0 = play, 1 = draw

for action in actions:
    old_observation = obs

    obs, reward, terminated, truncated, info = env.step(action)

    new_observation = obs

    print(f"\nAction {action}:")
    print(f"  Observation:  {old_observation} -> {new_observation}")
    print(f"  Reward:       {reward}")