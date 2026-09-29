# UNO Environment (`cs272/Uno-v0`)

This is a custom 2-player UNO environment for Gymnasium.

| Property | Value |
|---|---|
| Registration ID | `cs272/Uno-v0` |
| Action Space | `Discrete(2)` |
| Observation Space | `Discrete(160)` |

## Description

The game starts with each player drawing 7 cards from a shuffled 108-card deck. The player goes first.

Card values are encoded as 3-digit integers:
- `100` to `409`: Standard number cards (1st digit = color: 1=Red, 2=Blue, 3=Green, 4=Yellow).
- `x10`: Skip card
- `x20`: Reverse card (acts as Skip in 2-player mode)
- `x30`: Draw Two card
- `500`: Wild card (color change)
- `600`: Wild Draw Four card (color change + draw 4 + skip opponent)

## Actions

The action space is `Discrete(2)`:
- `0`: **Play** (Play the first playable card in hand matching middle card color/type or wild card)
- `1`: **Draw** (Draw a card from top of shuffled deck)

## Observations

Discrete state index in range `[0, 159]`:
$$\text{State} = (\min(\text{Hand Size}, 19) \times 4 + (\text{Color} - 1)) \times 2 + \text{Has Playable}$$

Total states: $20 \times 4 \times 2 = 160$. Hand size 19 represents 19 or more cards.

## Rewards

- **Win Game**: $+10.0$
- **Lose Game**: $-10.0$
- **Play Card**: $+1.0$
- **Draw Card**: $-1.0$

## Episode End

- **Terminated**: Player or dealer hand becomes empty.
- **Truncated**: Step count reaches 300 (`max_episode_steps=300`).