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
        410, 410, 420, 420, 430, 430
        ]


class UnoEnv(gym.Env):

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: str = "ansi"):

        self.render_mode = render_mode

        self.observation_space = spaces.Tuple(
            (   spaces.Box(
                low=100,
                high=430,
                shape=(),
                dtype=np.int32
                ),
                spaces.Box(
                low=-1,
                high=430,
                shape=(108,),
                dtype=np.int32
            ), 
            spaces.Discrete(108))
        )

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

        self.deck = deck.copy()
        self.np_random.shuffle(self.deck)
        self.dealer = [self.draw_card() for _ in range(7)]
        self.player = [self.draw_card() for _ in range(7)]
        self.middle_card = self.draw_card()

        while self.middle_card % 100 >= 10: #make sure initial middle card is not action card
            self.middle_card = self.draw_card()
        observation = self._get_obs()
        info = {}
        return observation, info

    def is_playable(self, card):

        card_color = card // 100
        card_type = card % 100

        middle_color = self.middle_card // 100
        middle_type = self.middle_card % 100

        return (card_color == middle_color 
                or card_type == middle_type)
    def dealer_turn(self):
        playable_cards = [card for card in self.dealer if self.is_playable(card)]
        if playable_cards:
            played_card = playable_cards[0]  # Play the first playable card
            self.dealer.remove(played_card)
            self.middle_card = played_card
        else:
            self.dealer.append(self.draw_card())

    def step(self, action):

        terminated = False
        truncated = False
        info = {}
        assert self.action_space.contains(action)

        if action: #draw a card (no card to play)
            self.player.append(self.draw_card())
            reward = -1.0
        else: #play a card (card to play)
            # Check if the player has a playable card
            playable_cards = [card for card in self.player if self.is_playable(card)]
            if playable_cards:
                played_card = playable_cards[0]  # Play the first playable card
                self.player.remove(played_card)
                self.middle_card = played_card
                reward = 1.0
            else:
                reward = -1.0  # No playable card, negative reward

        if len(self.player) == 0:
            reward = 10.0  # Player wins
            terminated = True

        #dealer's turn
        self.dealer_turn()

        if (len(self.dealer) == 0):
            reward = -10.0  # Dealer wins
            terminated = True

        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, info
    
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