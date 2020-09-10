import numpy as np
import gym
import pytest

import robo_gym




envs = [ 
    'EndEffectorPositioningUR10Sim-v0', 
    'EndEffectorPositioningUR10DoF5Sim-v0'
]


@pytest.mark.parametrize('env_name', envs)
def test_collision_detection(env_name):
    env = gym.make(env_name, ip='robot-servers')

    env.reset()
    done = False
    while not done:
        action = [1,1,1,1,1,1]
        _, _, done, info = env.step(action)

    assert info['final_status'] == 'collision'
