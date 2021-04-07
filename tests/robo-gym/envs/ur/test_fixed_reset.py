import numpy as np
import gym
import pytest

import robo_gym
from robo_gym.utils import ur_utils


test_ur_reset_init_joints_params = [
   ('EndEffectorPositioningUR5Sim-v0', [0.2, -2.5, 1.1, -2.0, -1.2, 1.2], 'ur5'),
   ('EndEffectorPositioningUR10Sim-v0', [0.2, -2.5, 1.1, -2.0, 1.2, 1.2], 'ur10')
]

@pytest.mark.parametrize('env_name, initial_joint_positions, ur_model', test_ur_reset_init_joints_params)
def test_ur_reset_init_joints(env_name, initial_joint_positions, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')

   state = env.reset(initial_joint_positions=initial_joint_positions)

   joint_comparison = np.isclose(ur.normalize_joint_values(initial_joint_positions), state[3:9], atol=0.1)

   for joint in joint_comparison:
      assert joint
   
   env.kill_sim()
   env.close()

test_ur_reset_desired_joint_positions_params = [
   ('MovingBoxTargetUR5Sim-v0', [0.5, -2.7, 1.3, -1.7, -1.9, 1.6], 'ur5'),
]

@pytest.mark.parametrize('env_name, desired_joint_positions, ur_model', test_ur_reset_desired_joint_positions_params)
def test_ur_reset_desired_joint_positions(env_name, desired_joint_positions, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')

   state = env.reset(desired_joint_positions=desired_joint_positions)

   joint_comparison = np.isclose(ur.normalize_joint_values(desired_joint_positions), state[3:9], atol=0.1)

   for joint in joint_comparison:
      assert joint
   
   env.kill_sim()
   env.close()