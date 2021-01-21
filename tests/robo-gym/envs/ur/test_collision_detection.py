import numpy as np
import gym
import pytest

import robo_gym




envs = [
    'EndEffectorPositioningUR10Sim-v0', 
    'EndEffectorPositioningUR10DoF5Sim-v0',
    'EndEffectorPositioningUR5Sim-v0',
    'EndEffectorPositioningUR5DoF5Sim-v0'
]


@pytest.mark.parametrize('env_name', envs)
def test_collision_detection(env_name):
    env = gym.make(env_name, ip='robot-servers')

    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        action = [1 for action in action]
        _, _, done, info = env.step(action)
        if done and info['final_status'] == 'success':
            env.reset()
            done = False
            

    assert info['final_status'] == 'collision'

    # env.kill_sim()
    # env.close()
