# UNO

This is a simplified UNO game supporting 2 players (user and the dealer) and UNO! (shout when you have 1 card left or face a penalty) feature coming soon.

|                   |                                   |
|-------------------|-----------------------------------|
| Make              | gymnasium.make("test/Uno-v0")    |
| Action Space      | Discrete(2)                      |
| Observation Space | Tuple(Box(100, 600, (), int32), Box(-1, 600, (108,), int32), Discrete(108)) |

UNO is a card game where the goal is to get rid of all your cards first, with each player starting with 7 cards each. 

## Description

The game starts with each player drawing 7 cards and there will be a middle starting card. All cards are drawn without replacement from a deck of 108 cards. The player goes first. There are 4 possible card colors (red, green, blue, yellow).

The card values are: 

- 1 set of cards numbered 0 - 9 (4 * each color)
- 1 set of cards numbered 1 - 9 (4 * each color)
- 1 set of wild cards (skip, reverse, draw 2) (4 * each color)
- 4 wild cards (color changing)
- 4 wild draw 2 cards (color changing)

The card values are encoded as a 3 digit integer with the first digit from the left as the color (1 = red, 2 = blue, 3 = green, 4 = yellow). The 2nd digit from the left is the card type (0 = normal digit cards, 1 = skip, 2 = reverse, 3 = draw two). The third digit is the corresponding numbers, for special card type they are always 0. 500 = wild cards. 600 = wild draw 2 cards.

The player must play a card that is either color matching or number matching to the middle card. 

If the player could not play a card, he must draw a card from the stack and his turn ends. 

The current logic for the wild card is automatically choosing the color based on the player's most number of color. (we hope to add this action of choosing color in the next iteration)

The game ends when either the dealer or the player has no more cards on their hand anymore.

## Starting state

Each player (dealer and agent) draws 7 cards each and there would be a middle starting card. The player goes first. 

## Actions

The action space is (1,) in the range {0,1} indicating whether to play or draw.

- 0: Draw
- 1: Play



## Observations

The obseraction space consists of a 3-tuple containing: the middle card, the current player's hand and dealer's number of cards. 

The observation is returned as (numpy.array(), numpy,array(108), int())

### Reward

- Win game: +10
- Lose game: -10
- Draw card: -1
- Play card: +1
- Skip player: 0

## Episode End

The episode ends if one of the following happens: 
- The player has no more cards left
- The dealer has no more cards left
- The stack of cards has run out

## Transition Noise

The deck is shuffled randomly at the beginning of each game. Cards drawn from the deck are random. The player receives a card from the top of the shuffled deck when choosing the draw action. The dealer also draws a top card from the shuffled deck when it does not have a playable card. Draw 2 and wild draw 4 cards cause additional cards from shuffled deck to be drawn. 

## Information

No additional information is returned.

## Version History

* v0: Initial versions release