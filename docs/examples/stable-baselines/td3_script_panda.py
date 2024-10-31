import gymnasium as gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1' # or other xxx.xxx.xxx.xxx

# initialize environment
env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=True)
# add wrapper for automatic exception handling
env = ExceptionHandling(env)

# follow the instructions provided by stable-baselines
model = TD3(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

# save model
model.save('td3_panda')