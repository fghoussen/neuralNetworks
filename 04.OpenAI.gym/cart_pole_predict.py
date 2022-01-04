# Import libraries.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # turn off tensor flow messages
tf.autograph.set_verbosity(3) # turn off tensor flow messages

import gym
import numpy as np
from cart_pole_utils import initialise, discretise_state, is_successful
import sys

# General parameters.

nb_episodes = 100
nb_state_divisions = 42

nn = True if len(sys.argv) == 2 and sys.argv[1] == 'nn' else False # Use neural networks or not.

# Functions.

def main():
    # Initialise.
    env = gym.make('CartPole-v1')
    low_state, step_state, _, q_table = initialise(env, nb_state_divisions, nn)
    if nn:
        q_table = tf.keras.models.load_model('cart_pole')
    else:
        q_table=np.load("cart_pole.npy")

    # Running successive episodes.
    for episode in range(nb_episodes):
        # Running one episode.
        state = env.reset()
        while True:
            # Render.
            env.render()

            # Discretise state.
            discret_state = discretise_state(state, low_state, step_state)

            # Use Q-table as we can trust it (after training).
            if nn:
                nb_samples, idx_sample, nb_states = 1, 0, len(state) # Neural network: used (for prediction) one state at a time.
                state_nn = np.asarray(state).reshape((nb_samples, nb_states))
                action = np.argmax(q_table.predict(state_nn)[idx_sample])
            else:
                action = np.argmax(q_table[discret_state])

            # Take action.
            state, reward, done, _ = env.step(action)

            # Checking new state.
            if done:
                break

        # Print progress.
        status = 'successful' if is_successful(state, env) else 'failed'
        print('episode %03d:'%episode, status, flush=True)

    # Close environment.
    env.close()

# Main program.

if __name__ == '__main__':
    main()
