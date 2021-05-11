<img align="left" width="60" height="60" src="https://user-images.githubusercontent.com/36470989/116858354-8668f380-abfe-11eb-81dc-629d9e8a9d4e.png" alt="robo-gym logo">

<!-- omit in toc -->
# robo-gym 

**robo-gym is an open source toolkit for distributed reinforcement learning on real and simulated robots.**

![](https://user-images.githubusercontent.com/36470989/81711550-11902d00-9474-11ea-8a04-d31da59e8266.gif) ![](https://user-images.githubusercontent.com/36470989/81711381-e73e6f80-9473-11ea-880e-1b0ff50e15ff.gif)

``robo-gym`` provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real world robotics. Additionally, we provide the tools to facilitate the creation of new environments featuring different robots and sensors.

<!-- TODO update  -->
Main features :
- [OpenAI Gym](https://gym.openai.com) interface for all the the environments
- **simulated** and **real** robots interchangeability, which enables a seamless transfer from training in simulation to application on the real robot.
- built-in **distributed** capabilities, which enable the use of distributed algorithms and distributed hardware
- based only on **open source** software, which allows to develop applications on own hardware and without incurring in cloud services fees or software licensing costs
- integration of 2 commercially available **industrial robots**: MiR 100, UR 10 (more to come)
- it has been successfully deployed to train a DRL algorithm to solve two different tasks in simulation that was able to solve the tasks on the real robots as well, without any further training in the real world

A [paper](https://arxiv.org/abs/2007.02753) describing robo-gym has been accepted for IROS 2020. A video showcasing the toolkit's
capabilities and additional info can be found on our [website](https://sites.google.com/view/robo-gym)

**NOTE**: We are continuously working to improve and expand robo-gym. If you are interested in reproducing the results obtained in the IROS 2020 paper please refer to v.0.1.0 for all the 3 repositories involved in the framework: [robo-gym](https://github.com/jr-robotics/robo-gym/tree/v0.1.0), [robo-gym-robot-servers](https://github.com/jr-robotics/robo-gym-robot-servers/tree/v0.1.0), [robo-gym-server-modules](https://github.com/jr-robotics/robo-gym-server-modules/tree/v0.1.0).

[See the News section](#news)

<!-- omit in toc -->
## Table of Contents

- [Basics](#basics)
- [Installation](#installation)
  - [Environment Side](#environment-side)
  - [Robot Server Side](#robot-server-side)
    - [Simplified Installation](#simplified-installation)
    - [Standard Installation](#standard-installation)
  - [Managing Multiple Python Versions](#managing-multiple-python-versions)
- [How to use](#how-to-use)
  - [Simulated Environments](#simulated-environments)
    - [Additional commands for Simulated Environments](#additional-commands-for-simulated-environments)
      - [restart simulation](#restart-simulation)
      - [kill simulation](#kill-simulation)
      - [Exception Handling Wrapper](#exception-handling-wrapper)
  - [Real Robot Environments](#real-robot-environments)
- [Environments](#environments)
- [Examples](#examples)
  - [Random Agent MiR100 Simulation Environment](#random-agent-mir100-simulation-environment)
- [Testing](#testing)
- [Contributing](#contributing)
- [News](#news)


## Basics
[back to top](#robo-gym)

The robo-gym framework is composed of several building blocks.
Detailed information on them is given [here](docs/the_framework.md) and in the [paper](https://arxiv.org/abs/2007.02753).

![robo-gym framework](https://user-images.githubusercontent.com/36470989/79330117-4498dc80-7f19-11ea-9de4-bed4f6390f3a.jpg)

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


## Installation
[back to top](#robo-gym)

### Environment Side
**Requirements:** robo-gym requires Python >= 3.5

You can perform a minimal install of robo-gym with:

```bash
git clone https://github.com/jr-robotics/robo-gym.git
cd robo-gym
pip install -e .
```
If you prefer, you can do a minimal install of the packaged version directly from PyPI:

```bash
pip install robo-gym
```

### Robot Server Side

The Robot Server Side can be installed on the same machine running the Environment Side
and/or on other multiple machines.

For the Robot Server Side there are two types of installation.

#### Simplified Installation

The Simplified Installation is intended for the users that want to use the provided
simulated environments as they come. The whole Robot Server Side is provided as
a Docker Container including Server Manager, Robot Servers, Command Handlers and
Simulated Robots allowing to get the standard robo-gym environments running with
minimal effort.

At the moment the Simplified Installation cannot be used with the Real Robots.

1. Install Docker following the [official documentation](https://docs.docker.com/get-docker/).

2. Execute the following command to pull and and start the Docker container provided:

```bash
run-rs-side-standard
```

The command is installed with the robo-gym installation, so make sure you have installed
robo-gym (Environment Side) before you try this out.


**NOTE**: At the moment the Simplified Installation does not support the visualization of the environments.
The gui option is not working.

#### Standard Installation

**Requirements:** The Standard Installation requires Ubuntu 16.04 or 18.04.

The Standard Installation is intended to be used with Real Robots,
for one or multiple Simulated Robots and for development purposes.


1. Install [robo-gym-robot-servers](https://github.com/jr-robotics/robo-gym-robot-servers)
following the instructions in the repository's README.

2. Install [robo-gym-server-modules](https://github.com/jr-robotics/robo-gym-server-modules)
for the system-wide Python 2.7 with:

```bash
pip install robo-gym-server-modules
```
<!-- omit in toc -->
##### Test the installation

To test the installation of *robo-gym-server-modules* try to run:  `start-server-manager` . 

If you get: `start-server-manager: command not found` it is most probably because your `$PATH` is not set correctly, to fix the problem add:

```bash
export PATH="/home/<your_username>/.local/bin:$PATH"
```

to your `.bashrc` file. 

In a second terminal window activate the Python 3.6 virtual environment and run:

```python
import gym, robo_gym

env = gym.make('NoObstacleNavigationMir100Sim-v0', ip='127.0.0.1', gui=True)

env.reset()
```

If you are running the ServerManager on a different PC replace *127.0.0.1* with the IP address of the machine. 

After running the command you should see the robot simulation starting and the initial state of the environment printed in the terminal window. 

Once you are done run `kill-server-manager` in a terminal window to kill the RobotServer and the ServerManager.

### Managing Multiple Python Versions

[Here](docs/managing_multiple_python_vers.md) you can find some additional information
on how to deal with multiple Python versions on the same machine.

## How to use
[back to top](#robo-gym)

The environments provided with robo-gym can be used in the same way of any other
OpenAI Gym environment. To get started is enough to run:

```python
import gym, robo_gym

# for a simulated robot environment
env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>')
# for a real robot environment
env = gym.make('EnvironmentNameRob-v0', rs_address='<robot_server_address>')

env.reset()
```

Each environment comes with a version to be run with a simulated version of the
robot and the scenario and version to be run with the real robot.
Simulated environments have a name ending with *Sim* whereas real robot environments
have a name ending with *Rob*.

### Simulated Environments

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

<!-- TODO add restart command and remove kill-server-manager  -->

To start the Server Manager it is necessary to make sure that
ROS and the robo-gym workspace are sourced with:

<!-- TODO remove this  -->
```bash
# Source ROS Melodic
source /opt/ros/melodic/setup.bash
# Source ROS Kinetic
# source /opt/ros/kinetic/setup.bash
source ~/robogym_ws/devel/setup.bash
```

It is then sufficient to run ``start-server-manager`` in the same shell.

The IP address of the machine on which the Server Manager is running has to
be passed as an argument to ``env.make``, if the Server Manager is running on the
same machine use ``ip='127.0.0.1'``.

By default the simulated environments are started in headless mode, without any graphical interface.

To start a simulated environment with **GUI** use the optional *gui* argument:

```python
env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>', gui=True)
```

#### Additional commands for Simulated Environments

The Simulation wrapper provides some extra functionalities to the Simulated Environments.

##### restart simulation

```python
env.restart_sim()
```

##### kill simulation

<!-- TODO replace with close -->
```python
env.kill_sim()
```

##### Exception Handling Wrapper

The Exception Handling Wrapper comes in handy when training on simulated environments.
The wrapper implements reaction strategies to common exceptions raised during training.
If one of the know exceptions is raised it tries to restart the Robot Server and the Simulation
to recover the system. If the exceptions happen during the reset of the environment the Robot Server
is simply restarted in the background, whereas, if exceptions happen during the execution of an
environment step the environment returns:

```python
return self.env.observation_space.sample(), 0, True, {"Exception":True, "ExceptionType": <Exception_type>}
```
Adding the wrapper to any simulated environment is very easy:

```python
import gym, robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling

env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>')
env = ExceptionHandling(env)
```

### Real Robot Environments

When making a real robot environment the Robot Server needs to be started manually,
once this is started, its address has to be provided as an argument to the ``env.make()``
method call.

## Environments 

See [List of Environments](docs/environments.md).

For information on creating your own environments, see [Creating your own Environments](docs/creating_environments.md).

## Examples
[back to top](#robo-gym)
### Random Agent MiR100 Simulation Environment
<!-- TODO change this to UR env -->
```python
import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling

target_machine_ip = '127.0.0.1' # or other machine 'xxx.xxx.xxx.xxx'

# initialize environment
env = gym.make('NoObstacleNavigationMir100Sim-v0', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)

num_episodes = 10

for episode in range(num_episodes):
    done = False
    while not done:
        # random step in the environment
        state, reward, done, info = env.step(env.action_space.sample())
```



Additional examples can be found [here](docs/examples)

## Testing 

<!-- TODO add export alias, run short test and long tests. Add link to this in installaiton section -->
## Contributing
[back to top](#robo-gym)

New environments and new robots and sensors implementations are welcome!

More details and guides on how to contribute will be added soon!

If you encounter troubles running robo-gym or if you have questions please submit a new [issue](https://github.com/jr-robotics/robo-gym/issues/new/).

## News
[back to top](#robo-gym)

- 2020-11-03
  + IROS 2020 is live! This year the event is on-demand and accessible for free to everyone. You can register at https://www.iros2020.org/ondemand/signup and find the presentation of our paper about robo-gym here https://www.iros2020.org/ondemand/episode?id=1357&id2=Transfer%20Learning&1603991207687

- 2020-07-07
  + The robo-gym paper has been accepted for IROS 2020 !
- 2020-06-02 (v0.1.7)
  + improved documentation
  + added exception handling feature to simulated environments

- 2020-04-27 (v0.1.1)
  + added Simplified Installation option for Robot Server Side

- 2020-04-15 (v0.1.0)
  + robo-gym first release is here!
