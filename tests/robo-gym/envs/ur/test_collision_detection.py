import numpy as np
import gym
import pytest

import robo_gym
from robo_gym.utils import ur_utils




test_ur_collision = [
   ('EndEffectorPositioningUR5Sim-v0', [0.0, -1.26, -3.14, 0.0, 0.0], 'ur5'), # self-collision
   ('EndEffectorPositioningUR5Sim-v0', [0.0, 1.0, 1.8, 0.0, 0.0], 'ur5'), # collision with ground 
   ('EndEffectorPositioningUR10Sim-v0', [0.0, -1.5, 3.14, 0.0, 0.0, 0.0], 'ur10'), # self-collision
   ('EndEffectorPositioningUR10Sim-v0', [0.0, 1.0, 1.15, 0.0, 0.0, 0.0], 'ur10')  # collision with ground 

]


@pytest.mark.parametrize('env_name, collision_joint_config, ur_model', test_ur_collision)
def test_collision_detection(env_name, collision_joint_config, ur_model):
    ur = ur_utils.UR(model=ur_model)
    env = gym.make(env_name, ip='robot-servers')
    env.reset()
    action = ur.normalize_joint_values(collision_joint_config)
    done = False
    while not done:
        _, _, done, info = env.step(action)    
    assert info['final_status'] == 'collision'
    env.kill_sim()
    env.close()


test_object_collision_avoidance_iros_params = [
   ('IrosEnv03UR5TrainingSim-v0', [-0.2, -0.1, 0.5], 'ur5')  
]

@pytest.mark.parametrize('env_name, fixed_object_position, ur_model', test_object_collision_avoidance_iros_params)
def test_object_collision_avoidance_iros(env_name, fixed_object_position, ur_model):
   ur = ur_utils.UR(model=ur_model)
   env = gym.make(env_name, ip='robot-servers')
   state = env.reset(fixed_object_position=fixed_object_position)
   done = False
   while not done:
       _, _, done, info = env.step(np.zeros(5))
   assert info['final_status'] == 'collision'
   env.kill_sim()
   env.close()