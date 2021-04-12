import numpy as np
import gym
import pytest

import robo_gym
from robo_gym.utils import ur_utils


test_object_coordinates_ee_pos_params = [
   #? robot up-right, target_coord_in_ee_frame 0.0, -0.3, 0.2, coordinates of target calculated using official dimensions from DH parameters. 
   #? first value is d4+d6
   #? second value is: d1+a2+a3+d5
   ('EndEffectorPositioningUR5Sim-v0', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.191 -0.2), (1.001 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571}, 'ur5'),  
   ('EndEffectorPositioningUR10Sim-v0', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.256 -0.2), (1.428 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur10')
   # ('ur3_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.194 -0.2), (0.692 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur3')
   # ('ur3e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.223 -0.2), (0.694 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur3e')
   # ('ur5e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.233 -0.2), (1.079 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur5e')
   # ('ur10e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.291 -0.2), (1.485 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur10e')
   # ('ur16e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.291 -0.2), (1.139 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur16e')

]

@pytest.mark.parametrize('env_name, joint_positions, object_coordinates, polar_coords,  ur_model', test_object_coordinates_ee_pos_params)
def test_object_coordinates_ee_pos(env_name, joint_positions, object_coordinates, polar_coords, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')
   state = env.reset(joint_positions=joint_positions, ee_target_pose=object_coordinates)

   
   assert np.isclose([polar_coords['r'], polar_coords['phi'], polar_coords['theta']], state[0:3], atol=0.1).all()
   
   env.kill_sim()
   env.close()

test_object_coordinates_avoidance_basic_params = [
   ('MovingBoxTargetUR5Sim-v0', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.191 + 0.2), (1.001 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571}, 'ur5')  
]

@pytest.mark.parametrize('env_name, desired_joint_positions, fixed_object_position, polar_coords,  ur_model', test_object_coordinates_avoidance_basic_params)
def test_object_coordinates_avoidance_basic(env_name, desired_joint_positions, fixed_object_position, polar_coords, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')
   state = env.reset(desired_joint_positions=desired_joint_positions, fixed_object_position=fixed_object_position)

   
   assert np.isclose([polar_coords['r'], polar_coords['phi'], polar_coords['theta']], state[0:3], atol=0.1).all()
   
   env.kill_sim()
   env.close()

test_object_coordinates_avoidance_iros_params = [
   ('IrosEnv03UR5TrainingSim-v0', [0.3, 0.1, 1.0], {'r': 0.980, 'theta': 2.488, 'phi': 1.246}, {'r': 0.607, 'theta': 1.023, 'phi': -0.825}, 'ur5')  
]

@pytest.mark.parametrize('env_name, fixed_object_position, polar_coords_ee, polar_coords_forearm, ur_model', test_object_coordinates_avoidance_iros_params)
def test_object_coordinates_avoidance_iros(env_name, fixed_object_position, polar_coords_ee, polar_coords_forearm, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')
   state = env.reset(fixed_object_position=fixed_object_position)

   
   assert np.isclose([polar_coords_ee['r'], polar_coords_ee['phi'], polar_coords_ee['theta']], state[0:3], atol=0.1).all()
   assert np.isclose([polar_coords_forearm['r'], polar_coords_forearm['phi'], polar_coords_forearm['theta']], state[22:25], atol=0.1).all()
   
   env.kill_sim()
   env.close()