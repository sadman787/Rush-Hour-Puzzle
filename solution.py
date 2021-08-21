#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
from search import *  # for search engines
from rushhour import *  # for Rush Hour specific classes and problems
import numpy as np
import scipy as sp


# RUSH HOUR GOAL TEST
def rushhour_goal_fn(state):
    # IMPLEMENT
    '''Have we reached a goal state?'''

    board_properties = state.get_board_properties()
    (board_size, goal_entrance, goal_direction) = (board_properties[0], board_properties[1], board_properties[2])

    for vs in state.get_vehicle_statuses():
        if vs[4]:
            length = vs[2]
            if (goal_direction == 'N' and vs[3] == False):
                return (vs[1] == goal_entrance)
            if (goal_direction == 'W' and vs[3] == True):
                return (vs[1] == goal_entrance)
            if goal_direction == 'S' and vs[3] == False:
                if goal_entrance[1] < (length - 1) :
                    y_temp = board_size[1] - (length - 1 - goal_entrance[1])
                else:
                    y_temp = goal_entrance[1] - (length - 1)
                return (vs[1][0] == goal_entrance[0] and vs[1][1] == y_temp)
            if (goal_direction == 'E' and vs[3] == True):
                if goal_entrance[0] < (length - 1):
                    x_temp = board_size[0] - (length - 1 - goal_entrance[0])
                else:
                    x_temp = goal_entrance[0] - (length - 1)
                return (vs[1][0] == x_temp and vs[1][1] == goal_entrance[1])
    return False


# RUSH HOUR HEURISTICS
def heur_zero(state):
    # IMPLEMENT
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0  # replace this


def heur_min_moves(state):
    # IMPLEMENT
    '''basic rushhour heuristic'''
    # An admissible heuristic is nice to have. Getting to the goal may require
    # many moves and each moves the goal vehicle one tile of distance.
    # Since the board wraps around, there are two different
    # directions that lead to the goal.
    # NOTE that we want an estimate of the number of ADDITIONAL
    #     moves required from our current state
    # 1. Proceeding in the first direction, let MOVES1 =
    #   number of moves required to get to the goal if it were unobstructed
    # 2. Proceeding in the second direction, let MOVES2 =
    #   number of moves required to get to the goal if it were unobstructed
    #
    # Our heuristic value is the minimum of MOVES1 and MOVES2 over all goal vehicles.
    # You should implement this heuristic function exactly, even if it is
    # tempting to improve it.

    board_properties = state.get_board_properties()
    (board_size, goal_direction) = (board_properties[0], board_properties[2])
    x_dist1, x_dist2, y_dist1, y_dist2 = (0, 0, 0, 0)

    for vs in state.get_vehicle_statuses():
        if vs[4]:
            length = vs[2]
            (x_v, y_v) = vs[1]
            (x_g, y_g) = state.get_board_properties()[1]

            # if state is in goal
            if (x_v == x_g) and (y_v == y_g) and (goal_direction == 'N' or goal_direction == 'W'):
                return 0

            # if goal is on right x axis of state
            if (x_g > x_v) and (y_g == y_v):
                y_dist1, y_dist2 = (0, 0)
                x_dist1 = abs(x_g - x_v)
                x_dist2 = x_v + 1 + (board_size[1] - 1 - x_g)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'W':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'E':
                    MOVES1 -= 1
                    MOVES2 += 1
                    return min(MOVES1, MOVES2)

            # if goal is on left x axis of state
            if (x_v > x_g) and (y_g == y_v):
                y_dist1, y_dist2 = (0, 0)
                x_dist1 = abs(x_v - x_g)
                x_dist2 = (board_size[1] - 1 - x_v) + x_g + 1
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'W':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'E':
                    MOVES1 += 1
                    MOVES2 -= 1
                    return min(MOVES1, MOVES2)

            # if goal is on top y axis of state
            if (y_v > y_g) and (x_v == x_g):
                x_dist1, x_dist2 = (0, 0)
                y_dist1 = abs(y_v - y_g)
                y_dist2 = y_g + 1 + (board_size[0] - 1 - y_v)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2
                if goal_direction == 'N':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'S':
                    MOVES1 += 1
                    MOVES2 -= 1
                    return min(MOVES1, MOVES2)

            # if goal is on bottom y axis of state
            if (y_g > y_v) and (x_v == x_g):
                x_dist1, x_dist2 = (0, 0)
                y_dist1 = abs(y_g - y_v)
                y_dist2 = y_v + 1 + (board_size[0] - 1 - y_g)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'N':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'S':
                    MOVES1 -= 1
                    MOVES2 += 1
                    return min(MOVES1, MOVES2)

            # if goal is on bottom right side of the state
            if (x_g > x_v) and (y_g > y_v):
                x_dist1 = abs(x_g - x_v)
                x_dist2 = x_v + 1 + (board_size[1] - 1 - x_g)
                y_dist1 = abs(y_g - y_v)
                y_dist2 = y_v + 1 + (board_size[0] - 1 - y_g)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'N':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'S':
                    MOVES1 -= 1
                    MOVES2 += 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'E':
                    MOVES1 -= 1
                    MOVES2 += 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'W':
                    return min(MOVES1, MOVES2)


            # if goal is on top right side of the state
            if (x_g > x_v) and (y_v > y_g):
                x_dist1 = abs(x_g - x_v)
                x_dist2 = x_v + 1 + (board_size[1] - 1 - x_g)
                y_dist1 = abs(y_v - y_g)
                y_dist2 = y_g + 1 + (board_size[0] - 1 - y_v)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'N':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'S':
                    MOVES1 += 1
                    MOVES2 -= 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'E':
                    MOVES1 -= 1
                    MOVES2 += 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'W':
                    return min(MOVES1, MOVES2)

            # if goal is on bottom left side of the state
            if (y_g > y_v) and (x_v > x_g):
                x_dist1 = abs(x_v - x_g)
                x_dist2 = (board_size[1] - 1 - x_v) + x_g + 1
                y_dist1 = abs(y_g - y_v)
                y_dist2 = y_v + 1 + (board_size[0] - 1 - y_g)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'N':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'S':
                    MOVES1 -= 1
                    MOVES2 += 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'E':
                    MOVES1 += 1
                    MOVES2 -= 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'W':
                    return min(MOVES1, MOVES2)

            # if goal is on top left side of the state
            if (y_v > y_g) and (x_v > x_g):
                x_dist1 = abs(x_v - x_g)
                x_dist2 = (board_size[1] - 1 - x_v) + x_g + 1
                y_dist1 = abs(y_v - y_g)
                y_dist2 = y_g + 1 + (board_size[0] - 1 - y_v)
                MOVES1 = x_dist1 + y_dist1
                MOVES2 = x_dist2 + y_dist2

                if goal_direction == 'N':
                    return min(MOVES1, MOVES2)
                if goal_direction == 'S':
                    MOVES1 += 1
                    MOVES2 -= 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'E':
                    MOVES1 += 1
                    MOVES2 -= 1
                    return min(MOVES1, MOVES2)
                if goal_direction == 'W':
                    return min(MOVES1, MOVES2)

    return float("inf")


def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a rush hour state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_min_moves has an obvious flaw.
    # Write a heuristic function that improves a little upon heur_min_moves to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    goal = state.get_board_properties()[1]
    goal = np.asarray(goal)


    temp = 0
    for vs in state.get_vehicle_statuses():
        if vs[4]:
            temp = vs[1]
            temp = np.asarray(temp)
            return np.linalg.norm(goal - temp)

    return float("inf")


def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a rush hour state)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    # Many searches will explore nodes (or states) that are ordered by their f-value.
    # For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    # You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    # The function must return a numeric f-value.
    # The value will determine your state's position on the Frontier list during a 'custom' search.
    # You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + (weight * sN.hval )


def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a rush hour state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''
    se = SearchEngine('custom', 'full')
    wrapped_fval_function = ( lambda sN : fval_function(sN, weight) )
    se.init_search(initial_state, rushhour_goal_fn, heur_fn, wrapped_fval_function)
    g_bound, h_bound, g_h_bound = (float("inf"), float("inf"), float("inf"))
    start = os.times()[0]
    final = se.search(timebound, (g_bound, h_bound, g_h_bound))
    end = os.times()[0]

    if final:
        g_h_bound = final.gval + heur_fn(final)
    else:
        return False

    remaining_time = timebound - (end - start)

    output = final
    while remaining_time > 0 and se.open.empty() == False:
        start = os.times()[0]
        new_final = se.search(remaining_time, (g_bound, h_bound, g_h_bound))
        end = os.times()[0]

        if new_final:
            g_h_bound = new_final.gval + heur_fn(new_final)
            output = new_final

        remaining_time = remaining_time - (end - start)

        return output


def anytime_gbfs(initial_state, heur_fn, timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a rush hour state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of anytime greedybfs algorithm'''

    se = SearchEngine('best_first', 'full')
    se.init_search(initial_state, rushhour_goal_fn, heur_fn)
    g_bound, h_bound, g_h_bound = (float("inf"), float("inf"), float("inf"))
    start = os.times()[0]
    final = se.search(timebound, (g_bound, h_bound, g_h_bound))
    end = os.times()[0]

    if final:
        g_bound = final.gval
    else:
        return False

    remaining_time = timebound - (end - start)

    output = final

    while remaining_time > 0 and se.open.empty() == False:
        start = os.times()[0]
        new_final = se.search(remaining_time, (g_bound, h_bound, g_h_bound))
        end = os.times()[0]

        if new_final:
            g_bound = new_final.gval
            output = new_final
        remaining_time = remaining_time - (end - start)

    return output
