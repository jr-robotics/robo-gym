import gym
import pytest

import robo_gym

envs = [
    'NoObstacleNavigationMir100Sim-v0', 
    'ObstacleAvoidanceMir100Sim-v0', 
    'EndEffectorPositioningUR10Sim-v0', 
    'EndEffectorPositioningUR10DoF5Sim-v0',
    'EndEffectorPositioningUR5Sim-v0'
]


@pytest.mark.parametrize('env_name', envs)
@pytest.mark.filterwarnings('ignore:UserWarning')
def test_env_initialization(env_name):
    env = gym.make(env_name, ip='robot-servers')

    env.reset()
    done = False
    for i in range(5):
        if not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

    assert env.observation_space.contains(observation)
