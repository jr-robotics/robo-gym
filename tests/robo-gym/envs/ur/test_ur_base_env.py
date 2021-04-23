import gym
import robo_gym
from robo_gym.utils import ur_utils
import math
import numpy as np 

import pytest

ur_models = [pytest.param('ur3', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur3e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur5', marks=pytest.mark.commit), \
             pytest.param('ur5e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur10', marks=pytest.mark.nightly), \
             pytest.param('ur10e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur16e', marks=pytest.mark.skip(reason='not implemented yet')), \
]

@pytest.fixture(autouse=True, scope='module', params=ur_models)
def env(request):
    env = gym.make('EmptyEnvironmentURSim-v0', ip='robot-servers', ur_model=request.param, fix_wrist_3=True)
    yield env
    env.kill_sim()
    env.close()

@pytest.mark.commit 
def test_initialization(env):
    env.reset()
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

    assert env.observation_space.contains(observation)

@pytest.mark.commit 
@pytest.mark.flaky(reruns=3)
def test_self_collision(env):
    collision_joint_config = {'ur5': [0.0, -1.26, -3.14, 0.0, 0.0], \
                              'ur10': [0.0, -1.5, 3.14, 0.0, 0.0]}
    env.reset()
    action = env.ur.normalize_joint_values(collision_joint_config[env.ur.model])
    done = False
    while not done:
        _, _, done, info = env.step(action)    
    assert info['final_status'] == 'collision'

@pytest.mark.commit 
@pytest.mark.flaky(reruns=3)
def test_collision_with_ground(env):
    collision_joint_config = {'ur5': [0.0, 1.0, 1.8, 0.0, 0.0], \
                              'ur10': [0.0, 1.0, 1.15, 0.0, 0.0]}
    env.reset()
    action = env.ur.normalize_joint_values(collision_joint_config[env.ur.model])
    done = False
    while not done:
        _, _, done, info = env.step(action)    
    assert info['final_status'] == 'collision'

@pytest.mark.commit     
def test_reset_joint_positions(env):
   joint_positions =  [0.2, -2.5, 1.1, -2.0, -1.2, 1.2]

   state = env.reset(base_joint_position = joint_positions[0], 
                    shoulder_joint_position = joint_positions[1],
                    elbow_joint_position = joint_positions[2],
                    wrist_1_joint_position = joint_positions[3], 
                    wrist_2_joint_position = joint_positions[4], 
                    wrist_3_joint_position = joint_positions[5])
   assert np.isclose(env.ur.normalize_joint_values(joint_positions), state[0:6], atol=0.1).all()


test_ur_fixed_joints = [
    ('EmptyEnvironmentURSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('EmptyEnvironmentURSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
]

@pytest.mark.nightly
@pytest.mark.parametrize('env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model', test_ur_fixed_joints)
@pytest.mark.flaky(reruns=3)
def test_fixed_joints(env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model):
    env = gym.make(env_name, ip='robot-servers', fix_base=fix_base, fix_shoulder=fix_shoulder, fix_elbow=fix_elbow, 
                                                fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2, fix_wrist_3=fix_wrist_3, ur_model=ur_model)
    state = env.reset()
    
    initial_joint_positions = state[0:6]

    
    # Take 20 actions
    action = env.action_space.sample()
    for _ in range(20):
        state, _, _, _ = env.step(action)
    
    joint_positions = state[0:6]

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