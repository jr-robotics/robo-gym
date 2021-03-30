import numpy as np
import gym
import pytest

import robo_gym
from robo_gym.utils import ur_utils


test_object_coordinates_params = [
   #? robot up-right, target_coord_in_ee_frame 0.0, -0.3, 0.2, coordinates of target calculated using official dimensions from CAD drawing.
   ('EndEffectorPositioningUR5Sim-v0', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (-0.192 -0.2), (1.001 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571}, 'ur5'),  
   ('EndEffectorPositioningUR10Sim-v0', [0.0, -1.57, 0.0, -1.57, 1.2, 1.2], [0.0, (-0.256 -0.2), (1.428 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur10')

]

@pytest.mark.parametrize('env_name, initial_joint_positions, object_coordinates, polar_coords,  ur_model', test_object_coordinates_params)
def test_object_coordinates(env_name, initial_joint_positions, object_coordinates, polar_coords, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')
   state = env.reset(initial_joint_positions=initial_joint_positions, ee_target_pose=object_coordinates)

   
   assert np.isclose([polar_coords['r'], polar_coords['phi'], polar_coords['theta']], state[0:3], atol=0.1).all()
   
   env.kill_sim()
   env.close()