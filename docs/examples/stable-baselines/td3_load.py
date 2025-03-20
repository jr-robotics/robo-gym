import gymnasium as gym
import robo_gym
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy

# specify the ip of the machine running the robot-server
target_machine_ip = "127.0.0.1"  # or other xxx.xxx.xxx.xxx

# initialize environment
env = gym.make("NoObstacleNavigationMir100Sim-v0", ip=target_machine_ip, gui=True)

model = TD3.load("td3_mir_basic")

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
            print(
                "Episode {} terminated after {} timesteps with reward {}".format(
                    passed_episodes, episode_timesteps, reward
                )
            )

    if passed_timesteps >= timesteps:
        break

env.close()
