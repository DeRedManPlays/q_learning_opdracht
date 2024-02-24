# ReadMe q_learning showcase <br>

Author: Ramon Reilman <br>
Date: 16-1-2024 <br>
version: 1.0 <br>

## Name <br>

q_learning_showcase/moduleren_opdracht <br>

## Description <br>

This program showcases q_learning, a form of reinforcement learning. <br>

An agent (blue circle) learns to explore it's environment. It also learns where to get food, and where it gets hurt <br>
this is teached to the agent by giving it rewards based on its position. <br>

I don't know what it could be used for, this is a very specific thing. <br>

The program contains 2 scripts: <br>

- main: This script contains the main game loop. <br>
- functies_qlearn: this script contains all of the functions that are used in the main. <br>

## Installation <br>

This program was built on python version 3.12.1, so python is needed. <br>
All libraries in the requirements.txt should be installed. <br>

## Usage <br>

To use this program: <br>

1. Put both scripts in to the same directory.
2. use the terminal, the script needs 2 arguments:
- If you want to use an empty q_table or load a previous one. (use: new or load)
- Amount of generations (in numbers)
3. use this command in the terminal: "python3 main.py new 1000"


After running the script, a pygame window should pop up. <br>
there are 3 states in this environment: <br>

1. -100 rewards (grey)
2. +200 rewards (green)
3. -5 rewards (white)

There is also a blue circle, this is the agent that explores the environment. <br>
after around 100-200 episodes is when the agent should consistent go to a green box. <br>

The position of terminal blocks are randomized, so if the agent is taking way longer to get to the green boxen, you can rerun the main. <br>

The agent sometimes freezes, its trying to find a new position that isn't terminal. <br>

## Support <br>

If any bugs are found:
mail r.reilman@st.hanze.nl <br>
for support.

## Acknowledgement

big thumbs up to everyone who helped with this project :) <br>
