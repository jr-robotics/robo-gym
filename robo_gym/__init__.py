from gym.envs.registration import register

# naming convention: EnvnameRobotSim

## Mir100 Environments
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

## UR5 Environments
register(
    id='EndEffectorPositioningUR5Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5Sim',
)

register(
    id='EndEffectorPositioningUR5Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5Rob',
)

register(
    id='EndEffectorPositioningUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5DoF5Sim',
)

register(
    id='EndEffectorPositioningUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR5DoF5Rob',
)

register(
    id='MovingBoxTargetUR5Sim-v0',
    entry_point='robo_gym.envs:MovingBoxTargetUR5Sim',
)

register(
    id='MovingBoxTargetUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:MovingBoxTargetUR5DoF3Sim',
)

register(
    id='MovingBoxTargetUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:MovingBoxTargetUR5DoF5Sim',
)

register(
    id='MovingBox3DSplineTargetUR5Sim-v0',
    entry_point='robo_gym.envs:MovingBox3DSplineTargetUR5Sim',
)

register(
    id='MovingBox3DSplineTargetUR5Rob-v0',
    entry_point='robo_gym.envs:MovingBox3DSplineTargetUR5Rob',
)

register(
    id='MovingBox3DSplineTargetUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:MovingBox3DSplineTargetUR5DoF3Sim',
)

register(
    id='MovingBox3DSplineTargetUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:MovingBox3DSplineTargetUR5DoF3Rob',
)

register(
    id='MovingBox3DSplineTargetUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:MovingBox3DSplineTargetUR5DoF5Sim',
)

register(
    id='MovingBox3DSplineTargetUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:MovingBox3DSplineTargetUR5DoF5Rob',
)

# 2 Boxes

register(
    id='Moving2Box3DSplineTargetUR5Sim-v0',
    entry_point='robo_gym.envs:Moving2Box3DSplineTargetUR5Sim',
)

register(
    id='Moving2Box3DSplineTargetUR5Rob-v0',
    entry_point='robo_gym.envs:Moving2Box3DSplineTargetUR5Rob',
)

register(
    id='Moving2Box3DSplineTargetUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:Moving2Box3DSplineTargetUR5DoF3Sim',
)

register(
    id='Moving2Box3DSplineTargetUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:Moving2Box3DSplineTargetUR5DoF3Rob',
)

register(
    id='Moving2Box3DSplineTargetUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:Moving2Box3DSplineTargetUR5DoF5Sim',
)

register(
    id='Moving2Box3DSplineTargetUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:Moving2Box3DSplineTargetUR5DoF5Rob',
)

# 1 Box 2 Points

register(
    id='ObstacleAvoidance1Box2PointsUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box2PointsUR5Sim',
)

register(
    id='ObstacleAvoidance1Box2PointsUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box2PointsUR5Rob',
)

register(
    id='ObstacleAvoidance1Box2PointsUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box2PointsUR5DoF3Sim',
)

register(
    id='ObstacleAvoidance1Box2PointsUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box2PointsUR5DoF3Rob',
)

register(
    id='ObstacleAvoidance1Box2PointsUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box2PointsUR5DoF5Sim',
)

register(
    id='ObstacleAvoidance1Box2PointsUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box2PointsUR5DoF5Rob',
)

# 2 Boxes 2 Points

register(
    id='ObstacleAvoidance2Box2PointsUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance2Box2PointsUR5Sim',
)

register(
    id='ObstacleAvoidance2Box2PointsUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance2Box2PointsUR5Rob',
)

register(
    id='ObstacleAvoidance2Box2PointsUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance2Box2PointsUR5DoF3Sim',
)

register(
    id='ObstacleAvoidance2Box2PointsUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance2Box2PointsUR5DoF3Rob',
)

register(
    id='ObstacleAvoidance2Box2PointsUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance2Box2PointsUR5DoF5Sim',
)

register(
    id='ObstacleAvoidance2Box2PointsUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance2Box2PointsUR5DoF5Rob',
)

## UR10 Environments
register(
    id='EndEffectorPositioningUR10Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10Sim',
)

register(
    id='EndEffectorPositioningUR10Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10Rob',
)

register(
    id='EndEffectorPositioningUR10DoF5Sim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10DoF5Sim',
)

register(
    id='EndEffectorPositioningUR10DoF5Rob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningUR10DoF5Rob',
)



