
robo-gym
********

**robo-gym is an open source toolkit for Distributed Deep Reinforcement Learning on real and simulated robots.**
The toolkit is build upon `OpenAI Gym <https://gym.openai.com>`_.

A paper describing robo-gym is currently submitted for IROS 2020. A video showcasing the toolkit's
capabilities and additional info can be found on our `website <https://sites.google.com/view/robo-gym>`_.

`See the News section <>`_

.. contents:: **Contents of this document**
   :depth: 2

Basics
======

The robo-gym framework is composed of several building blocks.
Detailed information on them is given in the paper.

..
.. .. image::
   :width: 25 %
   :align: center
   :alt: robo-gym framework
..

The framework can be subdivided in two main parts:

- The *Robot Server Side* (in green) is the one directly interacting with the real robot or
  simulating the robot. It is based on ROS Kinetic, Gazebo 7 and Python 2.7.
  It includes the robot simulation itself, the drivers to communicate with
  the real robot and additional required software tools.

- The *Environment Side* is the one providing the OpenAI Gym interface to the robot
  and implementing the different environments. It works with Python > 3.5.

The *Robot Server Side* and the *Environment Side* can run on the same PC or on different PCs
connected via network.
Given the different Python version requirements when running the *Robot Server Side*
and the *Environment Side* on the same PC, it is necessary to create two isolated
Python virtual environments.


Installation
============

Environment Side
----------------


robo-gym is provided as a package on the PyPI repository. You can install it with:

.. code-block:: shell

  pip install robo-gym

If you prefer you can also install it from source:

.. code-block:: shell

  git clone https://github.com/jr-robotics/robo-gym.git
  cd robo-gym
  pip install .


Robot Server Side
-----------------

First install `robo-gym-robot-servers <https://github.com/jr-robotics/robo-gym-robot-servers>`_
following the instructions in the repository's README.

After that install `robo-gym-server-modules <https://github.com/jr-robotics/robo-gym-server-modules>`_
for the system-wide Python 2.7 with:

.. code-block:: shell

  pip install robo-gym-server-modules




Environments
============

The environments provided with robo-gym can be used in the same way of any other
OpenAI Gym environment. To get started is enough to run:

.. code-block:: python

  import gym, robo_gym

  # for a simulated robot environment
  env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>')
  # for a real robot environment
  env = gym.make('EnvironmentNameRob-v0', rs_address='<robot_server_address>')

  env.reset()

Each environment comes with a version to be run with a simulated version of the
robot and the scenario and version to be run with the real robot.
Simulated environments have a name ending with *Sim* whereas real robot environments
have a name ending with *Rob*.

When making a simulated environment the ip address of the server manager needs to
be passed as an argument.

When making a real robot environment the Robot Server needs to be started manually,
once this is started, its address has to be provided as an argument to the ``env.make()``
method call.



Mobile Robots
-------------
Mobile Industrial Robots Mir100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Robot Arms
----------
Universal Robots UR10
~~~~~~~~~~~~~~~~~~~~~

``'EndEffectorPositioningUR10Sim-v0'``, ``'EndEffectorPositioningUR10Rob-v0'``

The goal in this environment is for the robotic arm to reach a target position with its end effector.

The target end effector positions are uniformly distributed across a semi-sphere of radius 1200 mm,
which is close to the full working area of the UR10.
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

``'EndEffectorPositioningAntiShakeUR10Sim-v0'``, ``'EndEffectorPositioningAntiShakeUR10Rob-v0'``

This environment has the same characteristics of *EndEffectorPositioningUR10Sim-v0* and
*EndEffectorPositioningUR10Rob-v0* with a different reward function.

The base reward that the agent receives at each step is proportional to the
variation of the three-dimensional Euclidean distance to the goal position.
Thus, a positive reward is received for moving closer to the goal, whereas a
negative reward is collected for moving away.
A penalty is given for high variation in the robot's joint velocities.
Both self collisions and collisions with the ground are taken into account and
punished with a negative reward and termination of the episode.

Examples
========

Examples and tutorials will be added soon!

News
====

- 2020-04-15 (v0.1.0)
  + robo-gym first release it's here!
