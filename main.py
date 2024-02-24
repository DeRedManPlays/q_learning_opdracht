"""
main.py

This script showcases a reinforcement learning algorithm called q-learning

Author: Ramon Reilman
version: 2.0
Date: 23-01-2024

to run:
python3 main.py new 1000
"""

import sys
from time import sleep
import pygame
import numpy as np
from functies_qlearn import drawgrid, drawterminal, is_terminal_state, \
                            gen_start_position, q_value_update, create_reward_nd_qtable, load_table


def main():
    save_or_load = sys.argv[1]
    generations = int(sys.argv[2])
    dt = 0
    # create color
    white = (255, 255, 255)

    # create the reward and qtable array
    reward, list_terminal = create_reward_nd_qtable()
    if save_or_load == "new":
        q_value = np.zeros((20, 20, 4))
    elif save_or_load == "load":
        q_value = load_table()

    # create the window space
    window_height = 400
    window_width = 400

    # initiate pygame
    pygame.init()

    # create the screen and clock
    SCREEN = pygame.display.set_mode((window_width, window_height))
    CLOCK = pygame.time.Clock()

    # define used items
    epsilon = 1
    epsilon_decay = 0.0001
    sleep_time = 0

    # set the window caption
    pygame.display.set_caption("Qlearning showcase")

    # main for loop, how many episodes the agent gets to learn
    for episode in range(generations):
        print(f"Episode: {episode}")

        # changes the epsilon and sleep_time
        if epsilon <= 0.8:
            epsilon = 1
        if episode == 1000:
            sleep_time = 0.05

        # gets first starting position
        positions = gen_start_position(reward)

        # main game while loop, stops when agent is on terminal position
        while not is_terminal_state(positions[0], positions[1], reward):

            # enables user to quit game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # fills background with white
            SCREEN.fill(white)

            # draws the env
            drawgrid(window_height, window_width, SCREEN)
            drawterminal(list_terminal, SCREEN, reward)

            # updates the q_value table
            q_value, positions = q_value_update(epsilon, q_value, reward, SCREEN, positions)

            # sleep to see agent move, and dt for framerate
            sleep(sleep_time)
            dt = CLOCK.tick(60) / 1000
        # epsilon slowly decays, pick the larger one
        epsilon = max(epsilon - epsilon_decay, 0)
    np.save("q_table_array", q_value)

if __name__ == "__main__":
    main()
