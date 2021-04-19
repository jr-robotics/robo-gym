from gym.envs.registration import register

# naming convention: EnvnameRobotSim

# MiR100 Environments
register(
    id='NoObstacleNavigationMir100Sim-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Sim',
)

register(
    id='NoObstacleNavigationMir100Rob-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Rob',
)

register(
    id='ObstacleAvoidanceMir100Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Sim',
)

register(
    id='ObstacleAvoidanceMir100Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Rob',
)

# UR Environments
register(
    id='EmptyEnvironmentURSim-v0',
    entry_point='robo_gym.envs:EmptyEnvironmentURSim',
)

register(
    id='EmptyEnvironmentURRob-v0',
    entry_point='robo_gym.envs:EmptyEnvironmentURRob',
)

register(
    id='EndEffectorPositioningURSim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningURSim',
)

register(
    id='EndEffectorPositioningURRob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningURRob',
)

register(
    id='MovingBoxTargetURSim-v0',
    entry_point='robo_gym.envs:MovingBoxTargetURSim',
)

register(
    id='MovingBoxTargetURRob-v0',
    entry_point='robo_gym.envs:MovingBoxTargetURRob',
)

# Iros Env 03
register(
    id='IrosEnv03URTrainingSim-v0',
    entry_point='robo_gym.envs:IrosEnv03URTrainingSim',
)

register(
    id='IrosEnv03URTrainingRob-v0',
    entry_point='robo_gym.envs:IrosEnv03URTrainingRob',
)

register(
    id='IrosEnv03URTestFixedSplinesSim-v0',
    entry_point='robo_gym.envs:IrosEnv03URTestFixedSplinesSim',
)

register(
    id='IrosEnv03URTestFixedSplinesRob-v0',
    entry_point='robo_gym.envs:IrosEnv03URTestFixedSplinesRob',
)




