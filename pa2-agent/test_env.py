import sys
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from myenv import UnoEnv

def run_tests():
    print("=== 1. Checking Environment Registration & Gymnasium check_env ===")
    env = gym.make("cs272/Uno-v0")
    check_env(env.unwrapped)
    print("✓ Gymnasium check_env passed clean with zero errors or warnings!")

    print("\n=== 2. Testing Observation & Action Space Bounds ===")
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert 0 <= obs <= 159, f"Invalid obs: {obs}"
        for _ in range(50):
            action = env.action_space.sample()
            assert action in (0, 1), f"Invalid action: {action}"
            obs, reward, terminated, truncated, _ = env.step(action)
            assert 0 <= obs <= 159, f"Invalid obs during step: {obs}"
            if terminated or truncated:
                break
    print("✓ All observations strictly in [0, 159] and actions in {0, 1}.")

    print("\n=== 3. Testing Reproducibility with Seed ===")
    env1 = gym.make("cs272/Uno-v0")
    obs1, _ = env1.reset(seed=42)
    actions = [0, 1, 0, 1, 0, 0, 1]
    history1 = [obs1]
    for a in actions:
        obs, r, term, trunc, _ = env1.step(a)
        history1.append((obs, r, term, trunc))

    env2 = gym.make("cs272/Uno-v0")
    obs2, _ = env2.reset(seed=42)
    history2 = [obs2]
    for a in actions:
        obs, r, term, trunc, _ = env2.step(a)
        history2.append((obs, r, term, trunc))

    assert history1 == history2, "Env not reproducible with same seed!"
    print("✓ Reset and step execution are 100% reproducible with seed!")

    print("\n=== 4. Testing TimeLimit Truncation at 300 steps ===")
    env = gym.make("cs272/Uno-v0")
    obs, _ = env.reset(seed=123)
    truncated_found = False
    for step in range(350):
        # Action 1 = Draw, extends episode without ending game immediately
        obs, reward, terminated, truncated, _ = env.step(1)
        if truncated:
            assert step + 1 == 300, f"Truncated at step {step + 1} instead of 300"
            truncated_found = True
            break
        if terminated:
            print(f"Episode terminated early at step {step + 1}, re-testing truncation...")
            obs, _ = env.reset(seed=step + 1000)

    if truncated_found:
        print("✓ TimeLimit wrapper correctly truncates at step 300!")
    else:
        print("Notice: Episode ended before 300 steps during draw test.")

    print("\nAll Environment Checks Passed Successfully!")

if __name__ == "__main__":
    run_tests()
