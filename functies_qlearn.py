"""
functies_qlearn.py

Contains all of the used functions in the main.py script

Author: Ramon Reilman
version: 1.0
Date: 23-01-2024

"""

import pygame
import numpy as np

def create_reward_nd_qtable():
    """ create_reward_nd_qtable
    This function creates the 2 used arrays, reward and q_value
    
    : param q_value is a 3d array that contains the best possible action for every position
    : param reward is an numpy array which has the value of every position in the game
    """
    # create list
    np.random.seed(6416)
    list_terminal = []

    # fill reward with all rewards
    reward = np.full((20, 20), -5)
    reward[0:, 0] = -100
    reward[10,10] = 200
    reward[0, 0:] = -100
    reward[0:, -1] = -100
    reward[-1, 0:] = -100

    # create 45 random terminal blocks
    for _ in range(45):
        x = np.random.randint(1,19)
        y = np.random.randint(1, 19)
        if reward[x,y] == -5:
            reward[x, y] = -100

    # create 5 random rewarding blocks
    for _ in range(5):
        x = np.random.randint(1,19)
        y = np.random.randint(1, 19)
        if reward[x,y] == -5:
            reward[x, y] = 200

    # add all terminal blocks to list_terminal
    for x in range(0, 20):
        for y in range(0,20):
            if reward[x, y] == -100 or reward[x, y] == 200:
                list_terminal.append((x,y))

    # create the q_table and fill it with 0

    return reward, list_terminal

def drawgrid(window_height, window_width, screen):
    """ drawGrid
    This function creates the grid layout in the pygame world.
    
    : param window_height is an int with the height of the window
    : param window_width is an int with the width of the window
    : param screen is the screen on which the grid gets drawn
    """
    # color and gridsize
    black = (200, 200, 200)
    blocksize = 20

    # for loop over the height and width of the screen
    for x in range(0, window_width, blocksize):
        for y in range(0, window_height, blocksize):

            # draws a line on the x and y spaces.
            rect = pygame.Rect(x, y, blocksize, blocksize)
            pygame.draw.rect(screen, black, rect, 1)

def drawterminal(list_terminal, screen, reward):
    """ drawTerminal
    This function draws all the terminal blocks in the game.
    Terminal blocks are spaces that kills the agent.
    
    : param list_terminal is a list that contains the x and y value
    of terminal block locations
    : param screen is the screen on which the grid gets drawn
    : param reward is an numpy array which has the value of every position in the game
    """
    # creates colors used
    black = (200, 200, 200)
    green = (0, 200, 0)

    # for loop through the list
    for x in list_terminal:

        # changes the position to x and y values that can be used in game
        pos_x = 20 *x[1]
        pos_y = 20 * x[0]

        # makes them black if they have a -100 reward
        if reward[x[0], [x[1]]] == -100:
            pygame.draw.rect(screen, black, (pos_x, pos_y, 20, 20))

        # makes block green when they are a positive reward
        elif reward[x[0], x[1]] == 200:
            pygame.draw.rect(screen, green, (pos_x, pos_y, 20, 20))


def gen_start_position(reward):
    """ gen_start_position
    creates x and y position for the agent to spawn on.
    only if x and y are not terminal.
    
    : param reward is an numpy array which has the value of every position in the game
    : return the row and column for reward/q_value array
    : return x_pos and y_pos for agent in pygame env
    """
    # generates x and y position
    start_row = np.random.randint(20)
    start_column = np.random.randint(20)

    # checks if they are not terminal
    while is_terminal_state(start_row, start_column, reward):
        start_row = np.random.randint(20)
        start_column = np.random.randint(20)

    # turn them into usable values.
    game_x_pos = 20 * start_column + 10
    game_y_pos = 20 * start_row + 10

    return start_row, start_column, game_x_pos, game_y_pos

def is_terminal_state(start_row, start_column, reward):
    """ is_terminal_state
    checks if position is on a terminal position or not
    
    : param start_row and start_column, int to check the reward on one position
    : param reward is an numpy array which has the value of every position in the game
    """
    # if reward is -5 then position is not terminal, return false
    if reward[start_row, start_column] == -5:
        return False
    else:
        return True

def get_next_action(row_i, column_i, epsilon, q_value):
    """ get_next_action
    looks at position, and generates an action based on said position
    
    : param row_i and column_i are the row and column in the q_value array
    : epsilon is an int thats used to see if the agent does a random action or not
    : q_value is a 3d array that contains the best possible action for every position
    
    : return an int 0-3
    """
    # when the random is smaller than epsilon return best action from q_table
    if np.random.random() < epsilon:
        return np.argmax(q_value[row_i, column_i])
    # else generate random action.
    else:
        return np.random.randint(4)

def get_next_location(current_row, current_colomn, action_index):
    """ get_next_location
    looks at the action_index and returns an action to take
    
    : param current row, current_column is the position in of the agent.
    : param action_index is an int between 0 and 4.
    
    : return new_row, new_column, pos_x, pos_y, ints to be used for position in pygame env.
    """
    # list of possible actions
    acties = ["up", "right", "down", "left"]
    new_row = current_row
    new_colomn = current_colomn

    # makes sure the agent cannot go out of bounds
    if acties[action_index] == "up" and current_row > 0:
        new_row -= 1
    elif acties[action_index] == "right" and current_colomn < 10 - 1:
        new_colomn += 1
    elif acties[action_index] == "down" and current_row < 10 - 1:
        new_row += 1
    elif acties[action_index] == "left" and current_colomn > 0:
        new_colomn -= 1

    # calculates usable positions
    pos_x = 20 *new_colomn + 10
    pos_y = 20 * new_row + 10

    return new_row, new_colomn, pos_x, pos_y

def q_value_update(epsilon, q_value, reward, screen, positions):
    """ q_value_update
    the function calculates the best action for the agent
    it also checks the reward the agent gets and updates the q_value table accordingly
    
    : epsilon is an int thats used to see if the agent does a random action or not
    : q_value is a 3d array that contains the best possible action for every position
    : param reward is an numpy array which has the value of every position in the game
    : param position is a list with the x and y positions, and linked row/column for the arrays.
    
    """
    # unpacks the position list
    row_i = positions[0]
    column_i = positions[1]
    x_pos = positions[2]
    y_pos = positions[3]

    # create color for agent
    blue = (0,0,255)

    # get the best possible action
    actie_index = get_next_action(row_i, column_i, epsilon, q_value)
    old_row, old_column = row_i, column_i


    # draw the circle agent.
    pygame.draw.circle(screen, blue, (x_pos, y_pos), 5)

    # get the next location best on action taken
    row_i, column_i, x_pos, y_pos = get_next_location(row_i, column_i, actie_index)

    # gets the old reward and q_value
    old_qvalue = q_value[old_row, old_column, actie_index]
    reward_gained = reward[row_i, column_i]

    # if the reward is good, let user know
    if reward_gained == 200:
        print('Found green box!')

    # calculate the temporal difference
    temporal_diff = reward_gained + (0.9 * np.max(q_value[row_i, column_i]))- old_qvalue

    # update q_table accordingly
    new_qvalue = old_qvalue + (0.9*temporal_diff)
    q_value[old_row, old_column, actie_index] = new_qvalue

    # update screen
    pygame.display.update()

    return q_value, (row_i, column_i, x_pos, y_pos)


def load_table():
    """ load_table
    Loads previously used q_table

    : return laoded q_table array
    """
    q_table = np.load("q_table_array.npy")
    return q_table