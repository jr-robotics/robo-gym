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


# 2 Boxes 2 Points

register(
    id='ObstacleAvoidance1Box1PointsVoxelOccupancyUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box1PointsVoxelOccupancyUR5Sim',
)

register(
    id='ObstacleAvoidance1Box1PointsVoxelOccupancyUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box1PointsVoxelOccupancyUR5Rob',
)

register(
    id='ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3Sim',
)

register(
    id='ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF3Rob',
)

register(
    id='ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5Sim',
)

register(
    id='ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidance1Box1PointsVoxelOccupancyUR5DoF5Rob',
)

# Fixed Trajectory Avoidance

register(
    id='FixedTraj1Box1PointsUR5Sim-v0',
    entry_point='robo_gym.envs:FixedTraj1Box1PointsUR5Sim',
)

register(
    id='FixedTraj1Box1PointsUR5Rob-v0',
    entry_point='robo_gym.envs:FixedTraj1Box1PointsUR5Rob',
)

register(
    id='FixedTraj1Box1PointsUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:FixedTraj1Box1PointsUR5DoF5Sim',
)

register(
    id='FixedTraj1Box1PointsUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:FixedTraj1Box1PointsUR5DoF5Rob',
)

# ? Variant B - Nice Environment with the position that the robot should keep that is changing over time. 

register(
    id='ObstacleAvoidanceVarB1Box1PointUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB1Box1PointUR5Sim',
)

register(
    id='ObstacleAvoidanceVarB1Box1PointUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB1Box1PointUR5Rob',
)

register(
    id='ObstacleAvoidanceVarB1Box1PointUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB1Box1PointUR5DoF3Sim',
)

register(
    id='ObstacleAvoidanceVarB1Box1PointUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB1Box1PointUR5DoF3Rob',
)

register(
    id='ObstacleAvoidanceVarB1Box1PointUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB1Box1PointUR5DoF5Sim',
)

register(
    id='ObstacleAvoidanceVarB1Box1PointUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB1Box1PointUR5DoF5Rob',
)

# ? Variant B 10 Points - Nice Environment with the position that the robot should keep that is changing over time. 

register(
    id='ObstacleAvoidanceVarB10Points1Box1PointUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB10Points1Box1PointUR5Sim',
)

register(
    id='ObstacleAvoidanceVarB10Points1Box1PointUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB10Points1Box1PointUR5Rob',
)

register(
    id='ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3Sim',
)

register(
    id='ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB10Points1Box1PointUR5DoF3Rob',
)

register(
    id='ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5Sim',
)

register(
    id='ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarB10Points1Box1PointUR5DoF5Rob',
)

# ? Variant B Trajecory 01 - Nice Environment with the position that the robot should keep that is changing over time. 

register(
    id='ObstacleAvoidanceVarBTraj011Box1PointUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarBTraj011Box1PointUR5Sim',
)

register(
    id='ObstacleAvoidanceVarBTraj011Box1PointUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarBTraj011Box1PointUR5Rob',
)

register(
    id='ObstacleAvoidanceVarBTraj011Box1PointUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarBTraj011Box1PointUR5DoF3Sim',
)

register(
    id='ObstacleAvoidanceVarBTraj011Box1PointUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarBTraj011Box1PointUR5DoF3Rob',
)

register(
    id='ObstacleAvoidanceVarBTraj011Box1PointUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarBTraj011Box1PointUR5DoF5Sim',
)

register(
    id='ObstacleAvoidanceVarBTraj011Box1PointUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarBTraj011Box1PointUR5DoF5Rob',
)

# ? Variant C - 3 different target points that the robot should reach while staying as close as 
# ? possible to the original trajectory

register(
    id='ObstacleAvoidanceVarCPickplace31Box1PointUR5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarCPickplace31Box1PointUR5Sim',
)

register(
    id='ObstacleAvoidanceVarCPickplace31Box1PointUR5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarCPickplace31Box1PointUR5Rob',
)

register(
    id='ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF3Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF3Sim',
)

register(
    id='ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF3Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF3Rob',
)

register(
    id='ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF5Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF5Sim',
)

register(
    id='ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF5Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceVarCPickplace31Box1PointUR5DoF5Rob',
)

# ? Iros Env 01 - 3 different target points that the robot should reach while staying as close as 
# ? possible to the original trajectory

register(
    id='IrosEnv01UR5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv01UR5Sim',
)

register(
    id='IrosEnv01UR5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv01UR5Rob',
)

register(
    id='IrosEnv01UR5DoF3Sim-v0',
    entry_point='robo_gym.envs:IrosEnv01UR5DoF3Sim',
)

register(
    id='IrosEnv01UR5DoF3Rob-v0',
    entry_point='robo_gym.envs:IrosEnv01UR5DoF3Rob',
)

register(
    id='IrosEnv01UR5DoF5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv01UR5DoF5Sim',
)

register(
    id='IrosEnv01UR5DoF5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv01UR5DoF5Rob',
)

# ? Iros Env 02 - 
# ? 3 different target points that the robot should reach while staying as close as 
# ? possible to the original trajectory
# ? 2 Points on robot
# ? 10% of cases fixed point

register(
    id='IrosEnv02UR5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5Sim',
)

register(
    id='IrosEnv02UR5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5Rob',
)

register(
    id='IrosEnv02UR5DoF3Sim-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5DoF3Sim',
)

register(
    id='IrosEnv02UR5DoF3Rob-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5DoF3Rob',
)

register(
    id='IrosEnv02UR5DoF5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5DoF5Sim',
)

register(
    id='IrosEnv02UR5DoF5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5DoF5Rob',
)

register(
    id='IrosEnv02UR5FixObstacleTrajDoF5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5FixObstacleTrajDoF5Sim',
)

register(
    id='IrosEnv02UR5FixObstacleTrajDoF5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv02UR5FixObstacleTrajDoF5Rob',
)

register(
    id='AblationTestEnvDoF5Sim-v0',
    entry_point='robo_gym.envs:AblationTestEnvDoF5Sim',
)
# ? Iros Env 03

register(
    id='IrosEnv03UR5TrainingDoF5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv03UR5TrainingDoF5Sim',
)

register(
    id='IrosEnv03UR5TrainingDoF5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv03UR5TrainingDoF5Rob',
)

# ? Test Environment with robot trajectories different from the ones on which it was trained.

register(
    id='IrosEnv03UR5TestDoF5Sim-v0',
    entry_point='robo_gym.envs:IrosEnv03UR5TestDoF5Sim',
)

register(
    id='IrosEnv03UR5TestDoF5Rob-v0',
    entry_point='robo_gym.envs:IrosEnv03UR5TestDoF5Rob',
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



