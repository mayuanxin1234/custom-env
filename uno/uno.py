'''
A simple custom Uno environment for Gymnasium.
Original Code from the Gymnasium documentation.
'''

from typing import Optional
import numpy as np
import gymnasium as gym

from gymnasium import spaces

#Red = 1, Blue = 2, Green = 3, Yellow = 4 (1st index)
#normal number card = 0, skip = 1, reverse = 2, draw two = 3 (2nd index)
#Each color has 1 set of 0 card, 2 sets of 1-9 cards, and 2 sets of action cards (Skip, Reverse, Draw Two)

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

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: str = "ansi"):

        self.render_mode = render_mode

        self.observation_space = spaces.Tuple(
            (   spaces.Box(
                low=100,
                high=600,
                shape=(),
                dtype=np.int32
                ),
                spaces.Box(
                low=-1,
                high=600,
                shape=(108,),
                dtype=np.int32
            ), 
            spaces.Discrete(108))
        )
        self.skip_player = False
        # Define what actions are available (2 actions, draw or play)
        self.action_space = gym.spaces.Discrete(2)

    def _get_obs(self):
        player_hand = self.player + [-1] * (108 - len(self.player))
        player_hand = np.array(player_hand, dtype=np.int32)

        return (
            np.array(self.middle_card, dtype=np.int32),
            player_hand,
            len(self.dealer)
        )
    def draw_card(self):
        return self.deck.pop()


    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.skip_player = False
        self.deck = deck.copy()
        self.np_random.shuffle(self.deck)
        self.dealer = [self.draw_card() for _ in range(7)]
        self.player = [self.draw_card() for _ in range(7)]
        self.middle_card = self.draw_card()

        while self.middle_card % 100 >= 10: #make sure initial middle card is not action card
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

        return (card_color == self.current_color 
                or card_type == middle_type)
    def choose_color(self, hand):

        color_counts = {
            1: 0,  # Red
            2: 0,  # Blue
            3: 0,  # Green
            4: 0   # Yellow
        }

        for card in hand:

            # Ignore Wild cards
            if card >= 500:
                continue

            color = card // 100

            if color in color_counts:
                color_counts[color] += 1

        return max(color_counts, key=color_counts.get)
    def apply_player_card_effect(self, played_card):

        skip_dealer = False

        # Wild
        if played_card == 500:

            self.current_color = self.choose_color(self.player)

        # Wild Draw Four
        elif played_card == 600:

            self.current_color = self.choose_color(self.player)

            # Dealer draws four
            for _ in range(4):
                self.dealer.append(self.draw_card())

            # Dealer loses turn
            skip_dealer = True

        else:

            # Normal colored card updates active color
            self.current_color = played_card // 100

            card_type = played_card % 100

            # Skip
            if card_type == 10:

                skip_dealer = True

            # Reverse
            elif card_type == 20:

                # With two players, Reverse works like Skip
                skip_dealer = True

            # Draw Two
            elif card_type == 30:

                for _ in range(2):
                    self.dealer.append(self.draw_card())

                skip_dealer = True

        return skip_dealer
    def apply_dealer_card_effect(self, played_card):

        # Wild
        if played_card == 500:
            self.current_color = self.choose_color(self.dealer)

        # Wild Draw Four
        elif played_card == 600:
            self.current_color = self.choose_color(self.dealer)

            for _ in range(4):
                self.player.append(self.draw_card())

            self.skip_player = True

        else:
            self.current_color = played_card // 100

            card_type = played_card % 100

            # Skip
            if card_type == 10:
                self.skip_player = True

            # Reverse
            elif card_type == 20:
                self.skip_player = True

            # Draw Two
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

            # NEW
            self.apply_dealer_card_effect(played_card)

        else:

            self.dealer.append(self.draw_card())

    def step(self, action):

        terminated = False
        truncated = False
        info = {}

        skip_dealer = False

        assert self.action_space.contains(action)

        if action:  # draw a card
            self.player.append(self.draw_card())
            reward = -1.0

        else:  # play a card

            playable_cards = [
                card
                for card in self.player
                if self.is_playable(card)
            ]

            if playable_cards:

                played_card = playable_cards[0]

                self.player.remove(played_card)

                self.middle_card = played_card

                reward = 1.0

                skip_dealer = self.apply_player_card_effect(
                    played_card
                )

            else:

                reward = -1.0

        if len(self.player) == 0:

            reward = 10.0
            terminated = True

        # Dealer gets a turn only if not skipped
        if not skip_dealer and not terminated:
            self.dealer_turn()

        if len(self.dealer) == 0:

            reward = -10.0
            terminated = True

        observation = self._get_obs()

        if self.skip_player:

            self.skip_player = False

            reward = 0.0

        # dealer gets another turn
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