from myenv import UnoEnv

from myagent import SarsaLambdaAgent

env = UnoEnv(render_mode="ansi")

agent = SarsaLambdaAgent(
            env,
            gamma=0.99,
            alpha=0.05,
            eps=0.1,
            lam=0.1,
            trace="accumulating",
            total_epi=5000,
            init_val=1.0,
            seed=42
        )

best_run = agent.best_run()

print(best_run)