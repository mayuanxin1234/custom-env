# CS272 Programming Assignment 2: UNO Custom Environment & SARSA(λ) Report

**Authors**: Devi & Ma Yuanxin  
**Environment Registration ID**: `cs272/Uno-v0`  
**GitHub Repository**: [mayuanxin1234/custom-env](https://github.com/mayuanxin1234/custom-env)  

---

## 1. Environment Description (`cs272/Uno-v0`)

### 1.1 Overview & Card Encoding
We implemented a 2-player custom Gymnasium environment for the card game UNO, played between an RL agent and an automated dealer strategy. The game utilizes a standard 108-card deck consisting of 4 colors (1 = Red, 2 = Blue, 3 = Green, 4 = Yellow) and special action cards. 

Cards are represented as 3-digit integers:
* **Number Cards (`100`–`409`)**: First digit is color ($1..4$), second digit is `0`, third digit is number ($0..9$).
* **Skip (`110`, `210`, `310`, `410`)**: Skips opponent turn.
* **Reverse (`120`, `220`, `320`, `420`)**: Functions as Skip in 2-player games.
* **Draw Two (`130`, `230`, `330`, `430`)**: Forces opponent to draw 2 cards and skips opponent turn.
* **Wild (`500`)**: Allows player to set the active color to their hand's most frequent color.
* **Wild Draw Four (`600`)**: Sets active color, forces opponent to draw 4 cards, and skips opponent turn.

---

### 1.2 Action Space & Observation Space

#### Action Space: `Discrete(2)`
* `0`: **Play** (plays the first valid matching card or wild card in hand).
* `1`: **Draw** (draws one card from top of the shuffled deck).

#### Observation Space: `Discrete(160)`
To keep tabular Q-learning scalable under the assignment's $<500$ state budget, we encoded the environment state into a single integer index $S \in [0, 159]$:

$$\text{State} = (\min(\text{player\_hand\_size}, 19) \times 4 + (\text{current\_color} - 1)) \times 2 + \text{has\_playable}$$

* $\text{hand\_size} \in [0, 19]$ (where 19 indicates 19 or more cards) $\rightarrow 20$ bins
* $\text{current\_color} \in [1, 4] \rightarrow 4$ colors
* $\text{has\_playable} \in \{0, 1\} \rightarrow 2$ states

$$\text{Total Discrete State Space} = 20 \times 4 \times 2 = 160 \text{ states}$$

---

### 1.3 Rewards & Episode Termination

* **Win Game**: $+10.0$ (when player hand becomes empty).
* **Lose Game**: $-10.0$ (when dealer hand becomes empty).
* **Play Card**: $+1.0$
* **Draw Card / Illegal Play**: $-1.0$
* **Episode Limit**: Truncated at **300 steps** via Gymnasium's `TimeLimit` wrapper.

---

## 2. Experimental Results & Analysis

### 2.1 Experimental Setup
* **Algorithms**: SARSA($\lambda$) with Accumulating Eligibility Traces vs. `RandomAgent` baseline.
* **Hyperparameters**: Learning rate $\alpha = 0.05$, Discount factor $\gamma = 0.99$, Exploration $\epsilon = 0.1$, Q-table initialization $Q_0(s,a) = 1.0$.
* **Runs**: 5 seeds ($0, 1, 2, 3, 4$) across 5 values of $\lambda \in \{0.0, 0.3, 0.6, 0.9, 1.0\}$ ($5 \times 5 = 25$ runs of 5,000 training episodes each).

---

### 2.2 Numerical Results Table

| $\lambda$ Decay | Episodes to Target Return ($\ge 1.0$) | Final 100-Episode Mean Return |
| :---: | :---: | :---: |
| **Random Baseline** | N/A | **-11.4940** |
| **$\lambda = 0.0$** | **100.0** | **6.0440** |
| **$\lambda = 0.3$** | **100.0** | **6.1540** |
| **$\lambda = 0.6$** | 102.8 | **6.2240** |
| **$\lambda = 0.9$** | 168.2 | **6.2660** |
| **$\lambda = 1.0$** | 309.8 | **3.1020** |

---

### 2.3 Discussion of Eligibility Traces & Credit Assignment

1. **Random Agent Comparison**: The `RandomAgent` baseline achieves a mean return of **-11.4940**, as unguided actions lead to draw penalties and losses. SARSA($\lambda$) achieves a peak return of **+6.2660**, proving that the agent successfully learns effective card-playing strategies.

2. **Impact of $\lambda$ on Convergence & Return**:
   * **$\lambda = 0.0$ (1-step SARSA)**: Updates only the immediate previous state-action pair $(S_t, A_t)$. While it converges rapidly to the target threshold (100 episodes) due to zero trace variance, credit for multi-step sequences propagates slowly.
   * **$\lambda = 0.9$ (Optimal Multi-step)**: Achieves the **highest overall final mean return (6.2660)**. Eligibility traces allow terminal win rewards ($+10.0$) to propagate multi-step backwards to early card selection decisions.
   * **$\lambda = 1.0$ (Monte Carlo Equivalent)**: Traces persist across the entire episode without geometric decay ($\gamma \lambda$). In a stochastic environment like UNO with random card draws, this introduces severe credit-assignment variance, slowing convergence (309.8 episodes to target) and lowering final performance (3.1020).

---

## 3. Sample Greedy Best Run Execution

* **Greedy Policy**: Trained $\lambda=0.9$ agent evaluated with $\epsilon=0$ (pure greedy).
* **Episode Outcome**: Clean Win in **78 steps**.
* **Undiscounted Return**: **+27.00**
* **Discounted Return ($\gamma=0.99$)**: **+16.46**

```text
Step  1 | State: 57 | Action: Play (0) | Reward: +1.0
Step  2 | State: 49 | Action: Play (0) | Reward: +1.0
...
Step 77 | State:  8 | Action: Play (0) | Reward: -1.0
Step 78 | State:  9 | Action: Play (0) | Reward: +10.0 (Win)
```
