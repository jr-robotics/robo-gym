import numpy as np
import gym
import pytest

import robo_gym

test_ur_reset = [
   ('EndEffectorPositioningUR10Sim-v0', [-4.04972017e-01, -6.76845312e-01, 1.19715083e+00,  1.91754410e-07, 1.42542467e-06, 4.86407465e-08]),
   ('EndEffectorPositioningUR10Sim-v0', [-7.47340143e-01, -1.55076730e+00, -1.61744893e+00, -2.05733500e-06, 4.71114618e-07, 1.05304405e-08])

]

@pytest.mark.parametrize('env_name, initial_joint_positions', test_ur_reset)
@pytest.mark.filterwarnings('ignore:UserWarning')
def test_ur_reset_init_joints(env_name, initial_joint_positions):
    env = gym.make(env_name, ip='robot-servers')

    state = env.reset(initial_joint_positions=initial_joint_positions)

    joint_comparison = np.isclose(initial_joint_positions, state[3:9], atol=0.1)

    for joint in joint_comparison:
        assert joint

