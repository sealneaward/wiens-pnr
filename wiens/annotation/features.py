import pandas as pd
from copy import copy
import numpy as np


def euclid_dist(location_1, location_2):
    """
    Get L2 distance from 2 difference sets of locations
    """
    distance = (location_1-location_2)**2
    distance = distance.sum(axis=-1)
    distance = np.sqrt(distance)
    return distance


def get_min_distance(movement, anno, target_1, target_2):
    """
    Get the min distance between targets
    over the entire pnr window

    Parameters
    ----------
    movement: pd.DataFrame
        SportVu location info
    anno: dict
        pnr anno information
    target_1: str
        player pnr role
    target_2: str
        player pnr role

    Returns
    -------
    min_dist: float
        singular distance value
    """
    target_1 = anno[target_1]
    target_1_movement = movement.loc[movement.player_id == target_1, ['x_loc', 'y_loc']].values
    if target_2 != 'hoop':
        target_2 = anno[target_2]
        target_2_movement = movement.loc[movement.player_id == target_2, ['x_loc', 'y_loc']].values
    else:
        target_2_movement = movement.loc[movement.player_id == target_1, ['hoop_loc_x', 'hoop_loc_y']].values
    pairwise_distances = euclid_dist(target_1_movement, target_2_movement)
    min_dist = np.min(pairwise_distances)

    return min_dist


def get_average_distance(movement, window, anno, target_1, target_2):
    """
    Get the average distance between targets
    over the specified pnr window

    Parameters
    ----------
    movement: pd.DataFrame
        SportVu location info
    window: str
        indicator of approach or execution window
    anno: dict
        info on pnr, when screen happens, etc
    target_1: str
        player pnr role
    target_2: str
        player pnr role

    Returns
    -------
    average_distance: float
        singular distance value
    """
    # TODO update for tfr
    if window == 'approach':
        movement = movement.loc[movement.game_clock >= anno['gameclock'], :]
    elif window == 'execution':
        movement = movement.loc[movement.game_clock <= anno['gameclock'], :]

    target_1 = anno[target_1]
    target_1_movement = movement.loc[movement.player_id == target_1, ['x_loc', 'y_loc']].values
    if target_2 != 'hoop':
        target_2 = anno[target_2]
        target_2_movement = movement.loc[movement.player_id == target_2, ['x_loc', 'y_loc']].values
    else:
        target_2_movement = movement.loc[movement.player_id == target_1, ['hoop_loc_x', 'hoop_loc_y']].values

    pairwise_distances = euclid_dist(target_1_movement, target_2_movement)
    sum = np.sum(pairwise_distances)
    game_clocks = movement['game_clock'].drop_duplicates(inplace=False).values
    screen_time_ind = np.argmin(np.abs(game_clocks - anno['gameclock']))

    if window == 'approach':
        average_distance = sum / np.abs(np.argmax(game_clocks) - screen_time_ind)
    elif window == 'execution':
        average_distance = sum / np.abs(screen_time_ind - np.argmin(game_clocks))

    return average_distance


def get_diff_distance(movement, window, anno, target_1, target_2):
    """
    Get the difference of distances between targets
    over the specified pnr window

    Parameters
    ----------
    movement: pd.DataFrame
        SportVu location info
    window: str
        indicator of approach or execution window
    anno: dict
        info on pnr, when screen happens, etc
    target_1: str
        player pnr role
    target_2: str
        player pnr role

    Returns
    -------
    diff_distance: float
        singular distance value of difference in window
    """
    # TODO update for tfr
    if window == 'approach':
        movement = movement.loc[movement.game_clock >= anno['gameclock'], :]
    elif window == 'execution':
        movement = movement.loc[movement.game_clock <= anno['gameclock'], :]

    target_1 = anno[target_1]
    target_1_movement = movement.loc[movement.player_id == target_1, ['x_loc', 'y_loc']].values
    if target_2 != 'hoop':
        target_2 = anno[target_2]
        target_2_movement = movement.loc[movement.player_id == target_2,  ['x_loc', 'y_loc']].values
    else:
        target_2_movement = movement.loc[movement.player_id == target_1, ['hoop_loc_x', 'hoop_loc_y']].values

    pairwise_distances = euclid_dist(target_1_movement, target_2_movement)

    game_clocks = movement['game_clock'].drop_duplicates(inplace=False).values
    screen_time_ind = np.argmin(np.abs(game_clocks - anno['gameclock']))
    screen_time = game_clocks[screen_time_ind]

    if window == 'approach':
        diff_distance = (pairwise_distances[0] - pairwise_distances[screen_time_ind]) / (game_clocks[0] - screen_time)
    elif window == 'execution':
        diff_distance = (pairwise_distances[screen_time_ind] - pairwise_distances[-1]) / (screen_time - game_clocks[-1])

    return diff_distance


def get_hoop_location(movement, anno):
    """
    Use hard logic to determine which half of court hoop is in.
    """
    game_clocks = movement['game_clock'].drop_duplicates(inplace=False).values
    screen_time_ind = np.argmin(np.abs(game_clocks - anno['gameclock']))
    screen_time = game_clocks[screen_time_ind]
    ball_loc_x = movement.loc[
        (movement.game_clock == screen_time) &
        (movement.player_id == -1),
        'x_loc'
    ].values[0]

    if ball_loc_x > 47:
        hoop_loc_x = 88.65
        hoop_loc_y = 25
    else:
        hoop_loc_x = 5.35
        hoop_loc_y = 25

    movement['hoop_loc_x'] = hoop_loc_x
    movement['hoop_loc_y'] = hoop_loc_y

    return movement