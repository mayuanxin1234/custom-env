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

print(f"Middle card: {obs[0]}, Number of player cards: {obs[1]}, Number of dealer cards: {obs[2]}")

# Test each action type
actions = [0, 1]  # 0 = play, 1 = draw

for action in actions:
    old_middle_card = obs[0]
    old_player_hand = [card for card in obs[1] if card != -1]
    old_dealer_number = obs[2]

    obs, reward, terminated, truncated, info = env.step(action)

    new_middle_card = obs[0]
    new_player_hand = [card for card in obs[1] if card != -1]
    new_dealer_number = obs[2]

    print(f"\nAction {action}:")
    print(f"  Middle card:  {old_middle_card} -> {new_middle_card}")
    print(f"  Player hand:  {old_player_hand} -> {new_player_hand}")
    print(f"  Dealer cards: {old_dealer_number} -> {new_dealer_number}")
    print(f"  Reward:       {reward}")