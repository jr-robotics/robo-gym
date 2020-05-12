
|logo| robo-gym
***************

.. |logo| image:: https://user-images.githubusercontent.com/36470989/81711567-15bc4a80-9474-11ea-8499-7adcd6ec96a7.png

**robo-gym is an open source toolkit for Distributed Deep Reinforcement Learning on real and simulated robots.**
The toolkit is built upon `OpenAI Gym <https://gym.openai.com>`_.

|obstavoid| |eepos|

.. |obstavoid| image:: https://user-images.githubusercontent.com/36470989/81711550-11902d00-9474-11ea-8a04-d31da59e8266.gif

.. |eepos| image:: https://user-images.githubusercontent.com/36470989/81711381-e73e6f80-9473-11ea-880e-1b0ff50e15ff.gif

A paper describing robo-gym is currently submitted for IROS 2020. A video showcasing the toolkit's
capabilities and additional info can be found on our `website <https://sites.google.com/view/robo-gym>`_.

`See the News section <https://github.com/jr-robotics/robo-gym#news>`_

.. contents:: **Contents of this document**
   :depth: 2

Basics
======

The robo-gym framework is composed of several building blocks.
Detailed information on them is given `here <docs/the_framework.md>`_.

.. image:: https://user-images.githubusercontent.com/36470989/79330117-4498dc80-7f19-11ea-9de4-bed4f6390f3a.jpg
   :alt: robo-gym framework


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
Python virtual environments. See the following section for further details.


Installation
============

Environment Side
----------------
**Requirements:** robo-gym requires Python >= 3.5

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

The Robot Server Side can be installed on the same machine running the Environment Side
and/or on other multiple machines.

For the Robot Server Side there are two types of installation.

Simplified Installation
~~~~~~~~~~~~~~~~~~~~~~~

The Simplified Installation is intended for the users that want to use the provided
simulated environments as they come. The whole Robot Server Side is provided as
a Docker Container including Server Manager, Robot Servers, Command Handlers and
Simulated Robots allowing to get the standard robo-gym environments running with
minimal effort.

At the moment the Simplified Installation cannot be used with the Real Robots.

1. Install Docker following the `official documentation <https://docs.docker.com/get-docker/>`_.

2. Execute the following command to pull and and start the Docker container provided:

.. code-block:: shell

  run-rs-side-standard

The command is installed with the robo-gym installation, so make sure you have installed
robo-gym (Environment Side) before you try this out.


**NOTE**: At the moment the Simplified Installation does not support the visualization of the environments.
The gui option is not working.

Standard Installation
~~~~~~~~~~~~~~~~~~~~~
**Requirements:** The Standard Installation requires a PC with Ubuntu 16.04.

The Standard Installation is intended to be used with Real Robots,
for one or multiple Simulated Robots and for development purposes.


1. Install `robo-gym-robot-servers <https://github.com/jr-robotics/robo-gym-robot-servers>`_
following the instructions in the repository's README.

2. Install `robo-gym-server-modules <https://github.com/jr-robotics/robo-gym-server-modules>`_
for the system-wide Python 2.7 with:

.. code-block:: shell

  pip install robo-gym-server-modules


Managing Multiple Python Versions
---------------------------------

`Here <docs/managing_multiple_python_vers.md>`_ you can find some additional information
on how to deal with multiple Python versions on the same machine.

How to use
==========

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

Simulated Environments
----------------------

Before making a simulated environment it is necessary to start the Server Manager.
Depending on the type of installation and setup that you chose the Server Manager
could be running on the same machine where you are calling ``env.make()`` or on
another machine connected via network.

The commands to control the Server Manager are:

- ``start-server-manager`` starts the Server Manager in the background
- ``attach-to-server-manager`` attaches the console to the Server Manager tmux session allowing to visualize the status of the Server Manager
- ``Ctrl+B, D`` detaches the console from the Server Manager tmux session
- ``kill-all-robot-servers`` kills all the running Robot Servers and the Server Manager
- ``kill-server-manager`` kills the Server Manager

To start the Server Manager it is necessary to make sure that
ROS and the robo-gym workspace are sourced with:

.. code-block:: shell

  source /opt/ros/kinetic/setup.bash
  source ~/robogym_ws/devel/setup.bash

It is then sufficient to run ``start-server-manager`` in the same shell.

The IP address of the machine on which the Server Manager is running has to
be passed as an argument to ``env.make``, if the Server Manager is running on the
same machine use ``ip='localhost'``.

By default the simulated environments are started in headless mode, without any graphical interface.

To start a simulated environment with **GUI** use the optional *gui* argument:

.. code-block:: python

  env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>', gui=True)

Real Robot Environments
-----------------------

When making a real robot environment the Robot Server needs to be started manually,
once this is started, its address has to be provided as an argument to the ``env.make()``
method call.

Environments
============

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

.. image:: https://user-images.githubusercontent.com/36470989/79962530-70bbdc80-8488-11ea-8999-d6db38e4264a.gif

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

.. image:: https://user-images.githubusercontent.com/36470989/79962368-3ce0b700-8488-11ea-83ac-c9e8995c2957.gif

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

Contributing
============

New environments and new robots and sensors implementations are welcome!

More details and guides on how to contribute will be added soon!

If you encounter troubles running robo-gym or if you have questions please submit a new issue.

News
====
- 2020-04-27 (v0.1.1)

  + added Simplified Installation option for Robot Server Side

- 2020-04-15 (v0.1.0)

  + robo-gym first release is here!
