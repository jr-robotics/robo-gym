import numpy as np
import gym
import pytest

import robo_gym
from robo_gym.utils import ur_utils




envs = ['IrosEnv03UR5TrainingSim-v0',
        'IrosEnv03UR5TestFixedSplinesSim-v0']


@pytest.mark.parametrize('env_name', envs)
def test_robot_trajectory_iros(env_name):
    ur = ur_utils.UR(model=ur_model)
    env = gym.make(env_name, ip='robot-servers')
    env.reset()
    action = np.zeros(5)
    joint_positions = [-1.23,-1.80,-1.07,-1.84,1.59, -0.07]
    for i in range(40):
        state, _, _, _ = env.step(action)

    assert np.isclose(ur.normalize_joint_values(joint_positions), state[3:9], atol=0.1).all()

    env.kill_sim()
    env.close()
