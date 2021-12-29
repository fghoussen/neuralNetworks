# Import libraries.

import gym
import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def initialise(env, nb_state_divisions):
    # Initialise Q-table (1 state = 1 position + 1 velocity).
    low_state = env.observation_space.low
    for idx, val in enumerate(low_state):
        if val < -5.: # Cut-off in case of big values.
            low_state[idx] = -5.
    high_state = env.observation_space.high
    for idx, val in enumerate(high_state):
        if val > 5.: # Cut-off in case of big values.
            high_state[idx] = 5.
    nb_positions = nb_state_divisions
    nb_velocities = nb_state_divisions
    nb_steps = np.array([nb_positions, nb_velocities])
    step_state = (high_state - low_state) / nb_steps
    print('states: low', low_state, ', high', high_state, ', step', step_state, flush=True)
    nb_actions = env.action_space.n
    q_table = np.random.uniform(low = -1., high = 1., size = (nb_positions, nb_velocities, nb_actions))
    return low_state, step_state, nb_actions, q_table

def discretise_state(state, low_state, step_state):
    # Discretise state.
    discret_state = (state - low_state) / step_state
    return tuple(discret_state.astype(np.int))

def is_successful(state, env):
    # Check if state is successful.
    return (state[0] >= env.goal_position) and (state[1] >= env.goal_velocity)
