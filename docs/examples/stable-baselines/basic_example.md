# Getting Started Using Stable-Baselines3

## Prerequisites
### Installation of robo-gym
Follow the installation instructions: https://github.com/jr-robotics/robo-gym#installation

### Installation of stable-baselines3: 
Follow the installation instructions: https://github.com/DLR-RM/stable-baselines3

## 1. Start the robot-server-manager
Start the robot-server-manager on the machine where the simulation is supposed to be run by entering the start command in the bash:

```bash
start-server-manager
```


## 2. Example of simulation training using the MiR100 base environment
Here is a quick example running TD3 in the MiR100 base environment

(_Python >=3.8_ / _robo-gym_ virtual environment)
```python
import gymnasium as gym
import robo_gym
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1'

# initialize environment (to render the environment set gui=True)
env = gym.make('NoObstacleNavigationMir100Sim-v0', ip=target_machine_ip, gui=False)

# follow the instructions provided by stable-baselines
model = TD3(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=15000)

# saving and loading a model
model.save('td3_mir_basic')
del model 
model = TD3.load("td3_mir_basic")

# reinitialize simulation environment with rendering enabled
env.kill_sim() 
env = gym.make('NoObstacleNavigationMir100Sim-v0', ip=target_machine_ip, gui=True)

# run the environment 10 times using the trained model
num_episodes = 10
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```