# Import libraries.

import gym
import numpy as np
from mountain_car_utils import initialise, discretise_state, is_successful

# General parameters.

nb_episodes = 25000
nb_state_divisions = 42
show_progress = 500 # Show progress every X episode.

# Bellman equation parameters.

alpha = 0.10 # Learning rate.
gamma = 0.98 # Discount rate.

# Exploration/exploitation parameters (epsilon greedy exploration).

epsilon = 1.
epsilon_min = 0.1
epsilon_min_reached = nb_episodes // 2

epsilon_decay = (epsilon - epsilon_min) / epsilon_min_reached # Linear decrease to a minimum.

# Functions.

def main():
    # Initialise.
    env = gym.make('MountainCar-v0')
    low_state, step_state, nb_actions, q_table = initialise(env, nb_state_divisions)

    # Running successive episodes.
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
    np.save("mountain_car", q_table)
    env.close()

# Main program.

if __name__ == '__main__':
    main()
