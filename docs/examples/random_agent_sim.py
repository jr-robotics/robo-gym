import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('NoObstacleNavigationMir100Sim-v0', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)

num_episodes = 10

for episode in range(num_episodes):
    done = False
    env.reset()
    while not done:
        # random step in the environment
        state, reward, done, info = env.step(env.action_space.sample())