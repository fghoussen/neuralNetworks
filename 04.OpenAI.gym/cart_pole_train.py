# Import libraries.

import gym
import numpy as np
from cart_pole_utils import initialise, discretise_state, is_successful
import sys
import random

# General parameters.

nb_episodes = 50000
nb_state_divisions = 42
show_progress = 1000 # Show progress every X episode.

nn = True if len(sys.argv) == 2 and sys.argv[1] == 'nn' else False # Use neural networks or not.
max_size_train = 64000 # Limit size of the training set (memory footprint).
batch_size_train = 64 # Train the neural network by batches of fixed size.
if nn: # On laptop without nvidia GPU, impossible to train neural network in decent times: network is not trained enough!...
    nb_episodes = 20 # We MUST reduce learning on architecture without GPU: results are BAD!...
    show_progress = 2

# Bellman equation parameters.

alpha = 0.10 # Learning rate.
gamma = 0.98 # Discount rate.

# Exploration/exploitation parameters (epsilon greedy exploration).

epsilon = 1.
epsilon_min = 0.1
epsilon_min_reached = nb_episodes // 2

epsilon_decay = (epsilon - epsilon_min) / epsilon_min_reached # Linear decrease to a minimum.

# Functions.

def train_neural_network(state_train, data, nb_states, nb_actions, q_table):
    # Train neural network.
    train_neural_network_dataset(state_train, data)
    batches = train_neural_network_select_batches(state_train)
    x_train, y_train = train_neural_network_prepare(nb_states, nb_actions, batches, q_table)
    train_neural_network_fit(q_table, x_train, y_train)

def train_neural_network_dataset(state_train, data):
    # Feed or renew training set.
    if len(state_train) < max_size_train: # Feed training set.
        state_train.append(data)
    else: # Renew training set.
        state_train.pop(0) # Remove oldest data.
        state_train.append(data)

def train_neural_network_select_batches(state_train):
    # Select random batches from training set to avoid over-fitting (close and correlated data).
    batches = None
    if batch_size_train > len(state_train):
        batches = random.sample(state_train, len(state_train))
    else:
        batches = random.sample(state_train, batch_size_train)
    return batches

def train_neural_network_prepare(nb_states, nb_actions, batches, q_table):
    # Create neural network inputs and outputs from batches.
    x_train = np.zeros((len(batches), nb_states))
    y_train = np.zeros((len(batches), nb_actions))
    for idx, batch in enumerate(batches):
        # Use neural network to make predictions.
        state, action, new_state, reward, done = batch
        max_futur_q, current_q = train_neural_network_prepare_predict(nb_states, new_state, state, q_table)

        # Implement Bellman equation as a gradient descent:
        #   Q_{k+1}(s, a) = Q_{k}(s, a) + alpha*(reward + gamma*max_{a'}[Q_{k}(s', a')] - Q_{k}(s, a))
        new_q = current_q # Training set target: reward + gamma*max_{a'}[Q_{k}(s', a')]
        if done:
            new_q[action] = reward # No next action to take if we are done.
        else:
            new_q[action] = reward + gamma*max_futur_q

        x_train[idx] = state
        y_train[idx] = new_q
    return x_train, y_train

def train_neural_network_prepare_predict(nb_states, new_state, state, q_table):
    # Use neural network to make predictions.
    nb_samples, idx_sample = 1, 0 # Neural network: used (for prediction) one state at a time.
    new_state_nn = np.asarray(new_state).reshape((nb_samples, nb_states))
    max_futur_q = np.max(q_table.predict(new_state_nn)[idx_sample])
    state_nn = np.asarray(state).reshape((nb_samples, nb_states))
    current_q = q_table.predict(state_nn)[idx_sample]
    return max_futur_q, current_q

def train_neural_network_fit(q_table, x_train, y_train):
    # Fit neural network data set.
    q_table.fit(x_train, y_train, epochs=1, verbose=0)

def main():
    # Initialise.
    env = gym.make('CartPole-v1')
    low_state, step_state, nb_actions, q_table = initialise(env, nb_state_divisions, nn)

    # Running successive episodes.
    state_train = [] # Neural network: training set.
    nb_success = 0
    for episode in range(nb_episodes):
        # Discretise initial state.
        state = env.reset()
        discret_state = discretise_state(state, low_state, step_state)

        # Running one episode.
        global epsilon
        done = False
        while not done:
            # Select action to take.
            action = None
            if np.random.random() > epsilon: # Exploitation: trust agent learned policy.
                if nn:
                    nb_samples, idx_sample, nb_states = 1, 0, len(state) # Neural network: used (for prediction) one state at a time.
                    state_nn = np.asarray(state).reshape((nb_samples, nb_states))
                    action = np.argmax(q_table.predict(state_nn)[idx_sample])
                else:
                    action = np.argmax(q_table[discret_state])
            else: # Exploration: try random action.
                action = np.random.randint(nb_actions)

            # Take action.
            new_state, reward, done, _ = env.step(action)
            new_discret_state = discretise_state(new_state, low_state, step_state)

            # Checking new state.
            if done:
                if is_successful(new_state, env):
                    nb_success += 1
                break

            # Apply Bellman equation.
            if nn:
                data = (state, action, new_state, reward, done)
                train_neural_network(state_train, data, len(state), nb_actions, q_table)
            else:
                max_futur_q = np.max(q_table[new_discret_state])
                current_qa = q_table[discret_state][action]
                new_q = (1. - alpha)*current_qa + alpha*(reward + gamma*max_futur_q)
                q_table[discret_state][action] = new_q

            # Update state.
            discret_state = new_discret_state

        # Update action policy.
        if epsilon >= epsilon_min:
            epsilon -= epsilon_decay

        # Print progress.
        if episode%show_progress == 0:
            print('episode {:05d}/{:05d}, success {:04d}/{:04d}, epsilon {:.3f}'.format(episode, nb_episodes, nb_success, show_progress, epsilon), flush=True)
            env.render()
            nb_success = 0

    # Save Q-table for later predictions and close environment.
    if nn:
        q_table.save('cart_pole')
    else:
        np.save("cart_pole", q_table)
    env.close()

# Main program.

if __name__ == '__main__':
    main()
