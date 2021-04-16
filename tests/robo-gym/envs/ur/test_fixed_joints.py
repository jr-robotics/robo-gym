import numpy as np
import gym
import pytest

import robo_gym
from robo_gym.utils import ur_utils
import math




test_ur_fixed_joints = [
    ('EmptyEnvironmentURSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('EndEffectorPositioningURSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('MovingBoxTargetURSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('IrosEnv03URTrainingSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('IrosEnv03URTestFixedSplinesSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3

    ('EmptyEnvironmentURSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
    ('EndEffectorPositioningURSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
    ('MovingBoxTargetURSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
    ('IrosEnv03URTrainingSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
    ('IrosEnv03URTestFixedSplinesSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow

    ('EmptyEnvironmentURSim-v0', False, False, False, False, False, True, 'ur10'), # fixed wrist_3
    ('EndEffectorPositioningURSim-v0', False, False, False, False, False, True, 'ur10'), # fixed wrist_3
    ('MovingBoxTargetURSim-v0', False, False, False, False, False, True, 'ur10'), # fixed wrist_3

    ('EmptyEnvironmentURSim-v0', True, False, True, False, False, False, 'ur10'), # fixed Base and Elbow
    ('EndEffectorPositioningURSim-v0', True, False, True, False, False, False, 'ur10'), # fixed Base and Elbow
    ('MovingBoxTargetURSim-v0', True, False, True, False, False, False, 'ur10'), # fixed Base and Elbow

]


@pytest.mark.parametrize('env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model', test_ur_fixed_joints)
@pytest.mark.flaky(reruns=3)
def test_fixed_joints(env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model):
    ur = ur_utils.UR(model=ur_model)
    env = gym.make(env_name, ip='robot-servers', fix_base=fix_base, fix_shoulder=fix_shoulder, fix_elbow=fix_elbow, 
                                                fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2, fix_wrist_3=fix_wrist_3, ur_model=ur_model)
    state = env.reset()
    
    initial_joint_positions = [0.0]*6
    if env_name == 'EmptyEnvironmentURSim-v0':
        initial_joint_positions = state[0:6]
    else:
        initial_joint_positions = state[3:9]
    
    # Take 20 actions
    action = env.action_space.sample()
    for _ in range(20):
        state, _, _, _ = env.step(action)
    
    joint_positions = [0.0]*6
    if env_name == 'EmptyEnvironmentURSim-v0':
        joint_positions = state[0:6]
    else:
        joint_positions = state[3:9]

    if fix_base:
        assert math.isclose(initial_joint_positions[0], joint_positions[0], abs_tol=0.05)
    if fix_shoulder:
        assert math.isclose(initial_joint_positions[1], joint_positions[1], abs_tol=0.05)
    if fix_elbow:
        assert math.isclose(initial_joint_positions[2], joint_positions[2], abs_tol=0.05)
    if fix_wrist_1:
        assert math.isclose(initial_joint_positions[3], joint_positions[3], abs_tol=0.05)
    if fix_wrist_2:
        assert math.isclose(initial_joint_positions[4], joint_positions[4], abs_tol=0.05)
    if fix_wrist_3:
        assert math.isclose(initial_joint_positions[5], joint_positions[5], abs_tol=0.05)

    env.kill_sim()
    env.close()


