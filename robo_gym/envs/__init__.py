# Example
from robo_gym.envs.example.example_env import ExampleEnvSim, ExampleEnvRob

# MiR100
from robo_gym.envs.mir100.mir100 import NoObstacleNavigationMir100Sim, NoObstacleNavigationMir100Rob
from robo_gym.envs.mir100.mir100 import ObstacleAvoidanceMir100Sim, ObstacleAvoidanceMir100Rob

# UR
from robo_gym.envs.ur.ur_base_env import EmptyEnvironmentURSim, EmptyEnvironmentURRob
from robo_gym.envs.ur.ur_ee_positioning import EndEffectorPositioningURSim, EndEffectorPositioningURRob
from robo_gym.envs.ur.ur_avoidance_basic import BasicAvoidanceURSim, BasicAvoidanceURRob
from robo_gym.envs.ur.ur_avoidance_raad import AvoidanceRaad2022URSim, AvoidanceRaad2022URRob
from robo_gym.envs.ur.ur_avoidance_raad import AvoidanceRaad2022TestURSim, AvoidanceRaad2022TestURRob

# Panda (ADDED BY FARHANG)
from robo_gym.envs.panda.panda_base_env import EmptyEnvironmentPandaSim, EmptyEnvironmentPandaRob
from robo_gym.envs.panda.panda_ee_positioning import EndEffectorPositioningPandaSim, EndEffectorPositioningPandaRob
from robo_gym.envs.panda.panda_follow_env import FollowPandaSim, FollowPandaRob
