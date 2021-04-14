import numpy as np
import gym
import pytest
import pathlib
import json

import robo_gym
from robo_gym.utils import ur_utils


robo_gym_path = pathlib.Path(__file__).parent.parent.parent.parent.parent.absolute().joinpath('robo_gym')




test_robot_trajectory_iros_params = [
    ('IrosEnv03UR5TrainingSim-v0', 'ur5', 'envs/ur5/robot_trajectories/trajectory_iros_2021.json'),
    ('IrosEnv03UR5TestFixedSplinesSim-v0', 'ur5', 'envs/ur5/robot_trajectories/trajectory_iros_2021.json')
    ]


@pytest.mark.parametrize('env_name, ur_model, traj_relative_path', test_robot_trajectory_iros_params)
def test_robot_trajectory_iros(env_name, ur_model, traj_relative_path):
    ur = ur_utils.UR(model=ur_model)
    env = gym.make(env_name, ip='robot-servers')
    env.reset()

    # load trajectory 
    traj_path = robo_gym_path.joinpath(traj_relative_path)
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
        assert np.isclose(ur.normalize_joint_values(traj_joint_positions), ur_j_pos_norm, atol=0.1).all()
        # check if calculation of delta joints is correct
        assert np.isclose((ur_j_pos_norm - desired_joints), delta_joints, atol=0.01).all()
    # check that the current trajectory point is a target point
    assert state[21] == 1.0
    # check that state machine has transitioned to segment 1 of trajectory 
    traj_joint_positions = trajectory[1][0]
    state, _, _, _ = env.step(action)
    assert np.isclose(ur.normalize_joint_values(traj_joint_positions), state[3:9], atol=0.1).all()

    env.kill_sim()
    env.close()
