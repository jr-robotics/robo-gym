from gym.envs.registration import register

# naming convention: EnvnameRobotSim

## Mir100 Environments
register(
    id='NoObstacleNavigationMir100Sim-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Sim',
    max_episode_steps=500
)

register(
    id='NoObstacleNavigationMir100Rob-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Rob',
    max_episode_steps=500
)

register(
    id='ObstacleAvoidanceMir100Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Sim',
    max_episode_steps=500
)

register(
    id='ObstacleAvoidanceMir100Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Rob',
    max_episode_steps=500
)

## UR10 Environments
register(
    id='EndEffectorPositioningUR10Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10Sim',
    max_episode_steps=500
)

register(
    id='EndEffectorPositioningUR10Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10Rob',
    max_episode_steps=500
)

register(
    id='EndEffectorPositioningAntiShakeUR10Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningAntiShakeUR10Sim',
    max_episode_steps=500
)

register(
    id='EndEffectorPositioningAntiShakeUR10Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningAntiShakeUR10Rob',
    max_episode_steps=500
)