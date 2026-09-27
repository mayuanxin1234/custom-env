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

The player must play a card that is either color matching or number matching to the middle card. 

If the player could not play a card, he must draw a card from the stack and his turn ends. 

The current logic for the wild card is automatically choosing the color based on the player's most number of color. (we hope to add this action of choosing color in the next iteration)

The game ends when either the dealer or the player has no more cards on their hand anymore.

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

The episode ends if the following happens: 
- The player has no more cards left
- The dealer has no more cards left
- The stack of cards has run out

## Information

No additional information is returned.

## Version History

* v0: Initial versions release