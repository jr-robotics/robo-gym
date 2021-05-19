import gym
import robo_gym
from robo_gym.utils import ur_utils
import math
import numpy as np 
import pathlib
import json

import pytest


robo_gym_path = pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute().joinpath('robo_gym')

ur_models = [pytest.param('ur3', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur3e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur5', marks=pytest.mark.commit), \
             pytest.param('ur5e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur10', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur10e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur16e', marks=pytest.mark.skip(reason='not implemented yet')), \
]

@pytest.fixture(scope='module', params=ur_models)
def env(request):
    env = gym.make('AvoidanceIros2021TestURSim-v0', ip='robot-servers', ur_model=request.param)
    env.request_param = request.param
    yield env
    env.kill_sim()

@pytest.mark.commit 
def test_initialization(env):
    assert env.ur.model == env.request_param
    env.reset()
    done = False
    env.step([0,0,0,0,0])
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

    assert env.observation_space.contains(observation)

@pytest.mark.skip(reason="This fails only in CI")
@pytest.mark.flaky(reruns=3)
def test_object_collision(env):
   params = {
       'ur5': {'object_coords':[-0.2, -0.1, 0.5], 'n_steps': 250},
   }
   
   env.reset(fixed_object_position=params[env.ur.model]['object_coords'])
   done = False
   i = 0
   while (not done) or i<=params[env.ur.model]['n_steps'] :
       _, _, done, info = env.step(np.zeros(5))
       i += 1
   assert info['final_status'] == 'collision'

@pytest.mark.commit     
def test_robot_trajectory(env):
    params = {
    'ur5': {'traj_relative_path':'envs/ur/robot_trajectories/trajectory_iros_2021.json'}  
    }

    env.reset()
    # load trajectory 
    traj_path = robo_gym_path.joinpath(params[env.ur.model]['traj_relative_path'])
    with open(traj_path) as json_file:
        trajectory = json.load(json_file)['trajectory']
 
    action = np.zeros(5)
    for i in range(len(trajectory[0])):
        traj_joint_positions = trajectory[0][i]
        state, _, _, _ = env.step(action)
        ur_j_pos_norm = state[3:9]
        delta_joints = state[9:15]
        desired_joints = state [18:24]
        # check if robot follows the trajectory in all steps of trajectory segment 0
        assert np.isclose(env.ur.normalize_joint_values(traj_joint_positions), ur_j_pos_norm, atol=0.1).all()
        # check if calculation of delta joints is correct
        assert np.isclose((ur_j_pos_norm - desired_joints), delta_joints, atol=0.01).all()
    # check that the current trajectory point is a target point
    assert state[24] == 1.0
    # check that state machine has transitioned to segment 1 of trajectory 
    traj_joint_positions = trajectory[1][0]
    state, _, _, _ = env.step(action)
    assert np.isclose(env.ur.normalize_joint_values(traj_joint_positions), state[3:9], atol=0.1).all()


test_ur_fixed_joints = [
    ('AvoidanceIros2021TestURSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('AvoidanceIros2021TestURSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
]

@pytest.mark.nightly
@pytest.mark.parametrize('env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model', test_ur_fixed_joints)
@pytest.mark.flaky(reruns=3)
def test_fixed_joints(env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model):
    env = gym.make(env_name, ip='robot-servers', fix_base=fix_base, fix_shoulder=fix_shoulder, fix_elbow=fix_elbow, 
                                                fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2, fix_wrist_3=fix_wrist_3, ur_model=ur_model)
    state = env.reset()
    initial_joint_positions = state[3:9] 
    # Take 20 actions
    action = env.action_space.sample()
    for _ in range(20):
        state, _, _, _ = env.step(action)
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