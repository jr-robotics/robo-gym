# Environments


 <!-- TODO add TOC -->

 
## Universal Robots 

Available UR models: UR3, UR3e, UR5, UR5e, UR10, UR10e, UR16

To select the robot model use: `ur_model='<ur3, ur3e, ur5, ur5e, ur10, ur10e, ur16e>'`
### End Effector Positioning

```python
# simulated robot environment
env = gym.make('EndEffectorPositioningURSim-v0', ur_model='ur10', ip='<server_manager_address>')
# real robot environment
env = gym.make('EndEffectorPositioningURRob-v0', ur_model='ur10', rs_address='<robot_server_address>')

```

![](https://user-images.githubusercontent.com/36470989/79962368-3ce0b700-8488-11ea-83ac-c9e8995c2957.gif)

The goal in this environment is for the robotic arm to reach a target position with its end effector.

The target end effector positions are uniformly distributed across a semi-sphere of the size close to the full working area of the robot.
Potential target points generated within the singularity areas of the working space are discarded.
The starting position is a random robot configuration.

The observations consist of 15 values: the spherical coordinates of the target
with the origin in the robot's base link, the six joint positions and the six joint velocities.

The robot uses position control; therefore, an action in the environment consists
of six normalized joint position values.

The base reward that the agent receives at each step is proportional to the
variation of the three-dimensional Euclidean distance to the goal position.
Thus, a positive reward is received for moving closer to the goal, whereas a
negative reward is collected for moving away.
Both self collisions and collisions with the ground are taken into account and
punished with a negative reward and termination of the episode.


 ! When resetting the Real Robot environment the robot could go in self collision, please be cautious. We are working on a solution to fix this.


## Mobile Industrial Robots Mir100

  <!--TODO change env names to line with gym.make   -->

``'NoObstacleNavigationMir100Sim-v0'``,  ``'NoObstacleNavigationMir100Rob-v0'``

In this environment, the task of the mobile robot is to reach a target position
in a obstacle-free environment.
At the initialization of the environment the target is randomly generated within a 2x2m area.
For the simulated environment the starting position of the robot is generated
randomly whereas for the real robot the last robot's position is used.

The observations consist of 4 values.
The first two are the polar coordinates of the target position in the robot's reference frame.
The third and the fourth value are the linear and angular velocity of the robot.

The action is composed of two values: the target linear and angular velocity of the robot.

The base reward that the agent receives at each step is proportional to the
variation of the two-dimensional Euclidean distance to the goal position.
Thus, a positive reward is received for moving closer to the goal, whereas a
negative reward is collected for moving away.
In addition, the agent receives a large positive reward for reaching the goal
and a large negative reward when crossing the external boundaries of the map.

``'ObstacleAvoidanceMir100Sim-v0'``, ``'ObstacleAvoidanceMir100Rob-v0'``

![](https://user-images.githubusercontent.com/36470989/79962530-70bbdc80-8488-11ea-8999-d6db38e4264a.gif)

In this environment, the task of the mobile robot is to reach a target position
without touching the obstacles on the way.
In order to detect obstacles, the MiR100 is equipped with two laser scanners,
which provide distance measurements in all directions on a 2D plane.
At the initialization of the environment the target is randomly placed on the
opposite side of the map with respect to the robot's position.
Furthermore, three cubes, which act as obstacles, are randomly placed in between
the start and goal positions. The cubes have an edge length of 0.5 m, whereas
the whole map measures 6x8 m.
For the simulated environment the starting position of the robot is generated
randomly whereas for the real robot the last robot's position is used.

The observations consist of 20 values.
The first two are the polar coordinates of the target position in the robot's reference frame.
The third and the fourth value are the linear and angular velocity of the robot.
The remaining 16 are the distance measurements received from the laser scanner
distributed evenly around the mobile robot.
These values were downsampled from 2\*501 laser scanner values to reduce the
complexity of the learning task.

The action is composed of two values: the target linear and angular velocity of the robot.

The base reward that the agent receives at each step is proportional to the
variation of the two-dimensional Euclidean distance to the goal position.
Thus, a positive reward is received for moving closer to the goal, whereas a
negative reward is collected for moving away.
In addition, the agent receives a large positive reward for reaching the goal
and a large negative reward in case of collision.

