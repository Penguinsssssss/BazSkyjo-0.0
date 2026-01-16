# BazSkyjo-0.0
Benjamin Sullivan Flex Friday project 2025/2026

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
