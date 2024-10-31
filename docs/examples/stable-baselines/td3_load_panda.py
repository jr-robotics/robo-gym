import gymnasium as gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy

# specify the ip of the machine running the robot-server
target_machine_ip = '172.27.171.120' # or other xxx.xxx.xxx.xxx

# initialize environment
env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=True)
# add wrapper for automatic exception handling
env = ExceptionHandling(env)

model = TD3.load('td3_panda')

timesteps = 1000

passed_timesteps = 0
passed_episodes = 0

while True:
    print("Starting episode {} - reset... ".format((passed_episodes + 1)))
    observation, info = env.reset()
    done = False
    episode_timesteps = 0
    print("Reset finished, entering execution.".format((passed_episodes + 1)))
    while not done:
        action, _ = model.predict(observation=observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        passed_timesteps += 1
        episode_timesteps += 1
        if passed_timesteps >= timesteps:
            break
        if done:
            passed_episodes += 1
            print("Episode {} terminated after {} timesteps with reward {}".format(passed_episodes, episode_timesteps, reward))

    if passed_timesteps >= timesteps:
        break

env.close()
