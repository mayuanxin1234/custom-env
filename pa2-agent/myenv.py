'''
A simple custom Uno environment for Gymnasium.
'''

from typing import Optional
import numpy as np
import gymnasium as gym

from gymnasium import spaces

# Red = 1, Blue = 2, Green = 3, Yellow = 4 (1st index)
# normal number card = 0, skip = 1, reverse = 2, draw two = 3 (2nd index)
# Each color has 1 set of 0 card, 2 sets of 1-9 cards, and 2 sets of action cards (Skip, Reverse, Draw Two)

deck = [ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 301, 302, 303, 304, 305, 306, 307, 308, 309, 
        400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 401, 402, 403, 404, 405, 406, 407, 408, 409,
        110, 110, 120, 120, 130, 130,
        210, 210, 220, 220, 230, 230,
        310, 310, 320, 320, 330, 330,
        410, 410, 420, 420, 430, 430,
        500, 500, 500, 500,
        600, 600, 600, 600
        ]


class UnoEnv(gym.Env):

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str = "ansi"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(160)
        self.skip_player = False
        # Define available actions: 0 = Play, 1 = Draw
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        hand_size = min(len(self.player), 19)
        color = self.current_color - 1
        has_playable = any(
            self.is_playable(card)
            for card in self.player
        )

        state = hand_size
        state = state * 4 + color
        state = state * 2 + int(has_playable)

        return int(state)

    def draw_card(self):
        if (self.deck is None) or (len(self.deck) == 0):
            self.deck = deck.copy()
            self.np_random.shuffle(self.deck)
        return self.deck.pop()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.skip_player = False
        self.deck = deck.copy()
        self.np_random.shuffle(self.deck)
        self.dealer = [self.draw_card() for _ in range(7)]
        self.player = [self.draw_card() for _ in range(7)]
        self.middle_card = self.draw_card()

        # Initial middle card must be a standard number card (< 500 and card % 100 < 10)
        while self.middle_card >= 500 or (self.middle_card % 100 >= 10):
            self.middle_card = self.draw_card()

        self.current_color = self.middle_card // 100
        observation = self._get_obs()
        info = {}
        return observation, info

    def is_playable(self, card):
        if card == 500 or card == 600:
            return True
        card_color = card // 100
        card_type = card % 100
        middle_type = self.middle_card % 100

        return (card_color == self.current_color or card_type == middle_type)

    def choose_color(self, hand):
        color_counts = {
            1: 0,  # Red
            2: 0,  # Blue
            3: 0,  # Green
            4: 0   # Yellow
        }

        for card in hand:
            if card >= 500:
                continue
            color = card // 100
            if color in color_counts:
                color_counts[color] += 1

        # Return color with max count, or fallback to Red (1) if hand only has wild cards
        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] == 0:
            return 1
        return max_color

    def apply_player_card_effect(self, played_card):
        skip_dealer = False

        # Wild
        if played_card == 500:
            self.current_color = self.choose_color(self.player)

        # Wild Draw Four
        elif played_card == 600:
            self.current_color = self.choose_color(self.player)
            for _ in range(4):
                self.dealer.append(self.draw_card())
            skip_dealer = True

        else:
            self.current_color = played_card // 100
            card_type = played_card % 100

            # Skip, Reverse (2-player), or Draw Two
            if card_type in (10, 20):
                skip_dealer = True
            elif card_type == 30:
                for _ in range(2):
                    self.dealer.append(self.draw_card())
                skip_dealer = True

        return skip_dealer

    def apply_dealer_card_effect(self, played_card):
        if played_card == 500:
            self.current_color = self.choose_color(self.dealer)

        elif played_card == 600:
            self.current_color = self.choose_color(self.dealer)
            for _ in range(4):
                self.player.append(self.draw_card())
            self.skip_player = True

        else:
            self.current_color = played_card // 100
            card_type = played_card % 100

            if card_type in (10, 20):
                self.skip_player = True
            elif card_type == 30:
                for _ in range(2):
                    self.player.append(self.draw_card())
                self.skip_player = True

    def dealer_turn(self):
        playable_cards = [
            card for card in self.dealer
            if self.is_playable(card)
        ]

        if playable_cards:
            played_card = playable_cards[0]
            self.dealer.remove(played_card)
            self.middle_card = played_card
            self.apply_dealer_card_effect(played_card)
        else:
            self.dealer.append(self.draw_card())

    def step(self, action):
        terminated = False
        truncated = False
        info = {}
        skip_dealer = False

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        if action == 1:  # Draw a card
            self.player.append(self.draw_card())
            reward = -1.0

        else:  # Play a card (action == 0)
            playable_cards = [
                card for card in self.player
                if self.is_playable(card)
            ]

            if playable_cards:
                played_card = playable_cards[0]
                self.player.remove(played_card)
                self.middle_card = played_card
                reward = 1.0
                skip_dealer = self.apply_player_card_effect(played_card)
            else:
                reward = -1.0

        # Check player win
        if len(self.player) == 0:
            reward = 10.0
            terminated = True

        # Dealer turn (if not skipped and player hasn't won)
        if not skip_dealer and not terminated:
            self.dealer_turn()
            if len(self.dealer) == 0:
                reward = -10.0
                terminated = True

            # If dealer played a card that skips the player, dealer plays again
            while self.skip_player and not terminated:
                self.skip_player = False
                self.dealer_turn()
                if len(self.dealer) == 0:
                    reward = -10.0
                    terminated = True

        observation = self._get_obs()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            print("--------------------")
            print("UNO")
            print("--------------------")
            print(f"Middle card: {self.middle_card}")
            print(f"Dealer cards: {len(self.dealer)}")
            print(f"Your cards ({len(self.player)}):")
            print(self.player)
            print("--------------------")
        else:
            return f"Middle card: {self.middle_card}, Dealer cards: {len(self.dealer)}, Your cards ({len(self.player)}): {self.player}"

    def close(self):
        pass


gym.register(
    id="cs272/Uno-v0",
    entry_point="myenv:UnoEnv",
    max_episode_steps=300,
)