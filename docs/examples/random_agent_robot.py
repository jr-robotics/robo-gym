import gymnasium as gym
import robo_gym

# PLEASE DON'T USE RANDOM SAMPLES WITH A REAL ROBOT! THIS IS JUST A CODE EXAMPLE!

robot_address = 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('NoObstacleNavigationMir100Rob-v0', rs_address=robot_address)

num_episodes = 1

for episode in range(num_episodes):
    done = False
    env.reset()
    while not done:
        # random step in the environment
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated