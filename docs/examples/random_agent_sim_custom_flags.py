import gymnasium as gym
import robo_gym
from robo_gym.envs.simulation_wrapper import Simulation

target_machine_ip = "127.0.0.1"  # or other machine 'xxx.xxx.xxx.xxx'

Simulation.verbose = True
Simulation.del_try_async_kill = False

# initialize environment
env = gym.make("NoObstacleNavigationMir100Sim-v0", ip=target_machine_ip, gui=True)

num_episodes = 10

for episode in range(num_episodes):
    done = False
    env.reset()
    while not done:
        # random step in the environment
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated

env.close()
