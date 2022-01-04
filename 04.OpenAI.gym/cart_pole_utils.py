# Import libraries.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # turn off tensor flow messages
tf.autograph.set_verbosity(3) # turn off tensor flow messages

import gym
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def initialise(env, nb_state_divisions, nn):
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
    nb_angles = nb_state_divisions
    nb_angle_rates = nb_state_divisions
    nb_steps = np.array([nb_positions, nb_velocities, nb_angles, nb_angle_rates])
    step_state = (high_state - low_state) / nb_steps
    print('states: low', low_state, ', high', high_state, ', step', step_state, flush=True)
    nb_actions = env.action_space.n
    q_table = None
    if nn: # q_table is a neural network that takes a state as input and produce action probabilities as outputs.
        q_table = create_neural_network(len(high_state), nb_actions)
    else: # q_table is an array.
        q_table = np.random.uniform(low = -1., high = 1., size = (nb_positions, nb_velocities, nb_angles, nb_angle_rates, nb_actions))
    return low_state, step_state, nb_actions, q_table

def discretise_state(state, low_state, step_state):
    # Discretise state.
    discret_state = (state - low_state) / step_state
    return tuple(discret_state.astype(np.int))

def is_successful(state, env):
    # Check if state is successful.
    return (np.abs(state[0]) < env.x_threshold) and (np.abs(state[2]) < env.theta_threshold_radians)

def create_neural_network(n_inputs, n_outputs):
    # Create neural network.
    x_inp = Input(shape=n_inputs)
    x_tmp = Dense(30, activation='relu', kernel_initializer='he_normal')(x_inp)
    x_tmp = Dense(30, activation='relu', kernel_initializer='he_normal')(x_tmp)
    x_out = Dense(n_outputs, activation='relu', kernel_initializer='he_normal')(x_tmp)
    q_table = Model(inputs=x_inp, outputs=x_out)
    q_table.compile(optimizer='adam', loss='mse') # Model used as regressor.
    return q_table
