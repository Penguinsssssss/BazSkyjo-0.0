# BazSkyjo-0.0
Benjamin Sullivan Flex Friday project 2025/2026

## Feb. 27
- Agent now learns very well
- Changed reward function to reward getting 2/3 of a set
- Changed replaybuffer error to only store np.array
- Changed activation function for advantage and value heads from relu -> linear
- Uploaded basic agent (avg score of 35)

## Feb. 13
- Spent whole day trying different combinations of reward functions and brain settings
  - 1 -> 2 -> 3 hidden layers
  - 64 -> 128 -> 256 neurons per layer
  - Number of opponents
  - REWARD FUNCTION
    - Treat unknown cards as avg of remaining unseen cards
    - Treat unknown cards as 12 (highest value)
    - Treat unknown cards as half of avg '*'THIS MADE PROGRESS'*'
    - Treat unknown cards as 99 (very big penalty per unknown)
    - Treat getting a row as 999 reward

## Feb. 6
- Finished implementation of skyjo game logic
- Added save/load
- Initial training of bot shows no improvement (100,000 episodes). Will play with both reward function and size of brain next week.

## Jan. 30
- Began impimenting Skjyo environment
- Began work on mutli-agent training
- Continued scritping on video

## Jan. 23
- Discovered issue with reward plateau
  - Reward fuctions incorrect
- Started work on scripting explanation video
- Not much time today, >1 hour speaker series

## Jan. 16
- A lot of technical difficulties today
  - Bot learns quickly but is unable to maintain intelligence when training extends over a long timeframe
- Fixed Quixx rules
  - Indexes for each of the spaces were shifted over by one
  - Green and Blue rows now count from 12 -> 2 (same as in the physical game)
- Changed output shape
  - Before there was one output for each of the squares, even if no possible combination of dice could mark that square
  - Changed output size from 45 -> 13, one output for each value of the dice faces
  - Changed output to take highest valued legal move, rather then award a penalty for attempting to play an illegal move

## Jan. 9
- Added Quixx_Env
  - Quixx is a strategy based dice game
  - Saw incredible results with input -> 128 -> 128 -> 32 -> dueling heads
    - Average score of -13 to average score of 0.5 over 200,000 episodes
    - Shows *definate* learning capabilities

## Dec. 5
- Made test rock paper scissors environment
- When I attemped to run it I ran into multiple array size michmatch errors
- Did not resolve them today, will work on them across next week

## Nov. 21
- Implimented backpropagation
- Implimented replay buffer
  - Both of these are untested as I do not yet have a working environment

## Nov. 14
- Started work on practice model
  - Object oriented
  - Can initialize with any number of layers and neurons per layer
  - Uses ReLU activation
  - Currently random weights and training data
- This is a nice resorce https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

## Oct. 31
- Added automatic scoring
- Started work on visuals, will be done by next Flex Friday

## Oct. 24
- Implimented base Skyjo game through use of some old code
- No visuals
- Manual scoring
