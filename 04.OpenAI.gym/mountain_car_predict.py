# Import libraries.

import gym
import numpy as np
from mountain_car_utils import initialise, discretise_state, is_successful

# General parameters.

nb_episodes = 100
nb_state_divisions = 42

# Functions.

def main():
    # Initialise.
    env = gym.make('MountainCar-v0')
    low_state, step_state, _, _ = initialise(env, nb_state_divisions)
    q_table=np.load("mountain_car.npy")

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
