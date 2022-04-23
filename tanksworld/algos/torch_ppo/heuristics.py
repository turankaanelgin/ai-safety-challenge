import math
import pdb

import torch
import numpy as np

from tanksworld.minimap_util import *


def distance_to_closest_enemy(state_vector, observation, num_agents=5):
    # Returns the distance to the closest enemy

    if len(state_vector) == 1: state_vector = state_vector[0]

    distances = []
    for tank_idx in range(num_agents):
        state = state_vector[tank_idx]
        x, y = state[0], -state[1]
        min_distance = np.infty

        for enemy_idx in range(5, 10):
            enemy_state = state_vector[enemy_idx]
            enemy_x, enemy_y = enemy_state[0], -enemy_state[1]
            dx = x - enemy_x
            dy = y - enemy_y

            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_distance:
                min_distance = dist

        enemy_channel = observation[:, tank_idx, 1]
        if torch.any(enemy_channel).item():
            distances.append(min_distance)
        else:
            distances.append(0.0)
    return distances


def get_enemy_heuristic(state_vector):
    # Orient towards your closest enemy

    if len(state_vector) == 1: state_vector = state_vector[0]

    heuristic_actions = []
    for tank_idx in range(5):
        state = state_vector[tank_idx]
        x, y = state[0], -state[1]
        heading = state[2]
        min_distance = np.infty
        min_angle = 0

        all_enemy_points = []
        for enemy_idx in range(5, 10):
            enemy_state = state_vector[enemy_idx]
            enemy_x, enemy_y = enemy_state[0], -enemy_state[1]
            dx = x - enemy_x
            dy = y - enemy_y

            rel_enemy_x, rel_enemy_y = point_relative_point_heading([enemy_x, enemy_y], [x, y], heading)
            rel_enemy_x = (rel_enemy_x / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
            rel_enemy_y = (rel_enemy_y / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
            all_enemy_points.append((rel_enemy_x, rel_enemy_y))

            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)
            if dist < min_distance:
                min_distance = dist
                min_angle = abs(angle)
                min_enemy_x = rel_enemy_x
                min_enemy_y = rel_enemy_y
                if rel_enemy_x < 64:
                    orient_coeff = -1
                else:
                    orient_coeff = 1
                if rel_enemy_y > 64:
                    translate_coeff = -1
                else:
                    translate_coeff = 1

        #if tank_idx == 0:
        #    print('MIN_ENEMY_X_Y', [min_enemy_x, min_enemy_y])

        #if tank_idx == 0:
        heuristic_action = np.asarray(([[translate_coeff*0.5, orient_coeff * (min_angle % 1), -1.0]]))
        #else:
        #heuristic_action = np.zeros((1,3))
        heuristic_actions.append(heuristic_action)

    return np.expand_dims(np.concatenate(heuristic_actions, axis=0), axis=0)


def get_ally_heuristic_2(state_vector):
    # Orient against your closest ally to encourage exploration

    heuristic_actions = []
    for tank_idx in range(5):
        state = state_vector[tank_idx]
        x, y = state[0], -state[1]
        heading = state[2]
        min_distance = np.infty
        min_angle = 0

        all_ally_points = []
        for ally_idx in range(5):
            if ally_idx != tank_idx:
                ally_state = state_vector[ally_idx]
                ally_x, ally_y = ally_state[0], -ally_state[1]
                dx = x - ally_x
                dy = y - ally_y

                rel_ally_x, rel_ally_y = point_relative_point_heading([ally_x, ally_y], [x, y], heading)
                rel_ally_x = (rel_ally_x / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                rel_ally_y = (rel_ally_y / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                all_ally_points.append((rel_ally_x, rel_ally_y))

                dist = math.sqrt(dx * dx + dy * dy)
                angle = math.atan2(dy, dx)
                if dist < min_distance:
                    min_distance = dist
                    min_angle = angle
                    if rel_ally_x < 64:
                        orient_coeff = 1
                    else:
                        orient_coeff = -1
                    if rel_ally_y > 64:
                        translate_coeff = 1
                    else:
                        translate_coeff = -1

        heuristic_action = np.asarray(([[translate_coeff * 0.5, orient_coeff * min_angle, -1.0]]))
        heuristic_actions.append(heuristic_action)

    return np.expand_dims(np.concatenate(heuristic_actions, axis=0), axis=0)


def get_ally_heuristic(state_vector):
    # Orient towards your closest ally

    import matplotlib.pyplot as plt

    heuristic_actions = []
    for tank_idx in range(5):
        state = state_vector[tank_idx]
        x, y = state[0], -state[1]
        heading = state[2]
        min_distance = np.infty
        min_angle = 0

        all_ally_points = []
        for ally_idx in range(5):
            if ally_idx != tank_idx:
                ally_state = state_vector[ally_idx]
                ally_x, ally_y = ally_state[0], -ally_state[1]
                dx = x - ally_x
                dy = y - ally_y

                rel_ally_x, rel_ally_y = point_relative_point_heading([ally_x, ally_y], [x, y], heading)
                rel_ally_x = (rel_ally_x / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                rel_ally_y = (rel_ally_y / UNITY_SZ) * SCALE + float(IMG_SZ) * 0.5
                all_ally_points.append((rel_ally_x, rel_ally_y))

                dist = math.sqrt(dx * dx + dy * dy)
                angle = math.atan2(dy, dx)
                if dist < min_distance:
                    min_distance = dist
                    min_angle = angle
                    if rel_ally_x < 64:
                        orient_coeff = -1
                    else:
                        orient_coeff = 1
                    if rel_ally_y > 64:
                        translate_coeff = -1
                    else:
                        translate_coeff = 1

        heuristic_action = np.asarray(([[translate_coeff * 0.5, orient_coeff * min_angle, -1.0]]))
        heuristic_actions.append(heuristic_action)

    return np.expand_dims(np.concatenate(heuristic_actions, axis=0), axis=0)