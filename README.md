<img align="left" width="60" height="60" src="https://user-images.githubusercontent.com/36470989/116858354-8668f380-abfe-11eb-81dc-629d9e8a9d4e.png" alt="robo-gym logo">

<!-- omit in toc -->
# robo-gym 

<br>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/robo-gym)
![PyPI](https://img.shields.io/pypi/v/robo-gym)

**``robo-gym`` is an open source toolkit for distributed reinforcement learning on real and simulated robots.**

![](https://user-images.githubusercontent.com/36470989/81711550-11902d00-9474-11ea-8a04-d31da59e8266.gif) ![](https://user-images.githubusercontent.com/36470989/81711381-e73e6f80-9473-11ea-880e-1b0ff50e15ff.gif)

``robo-gym`` provides a collection of reinforcement learning environments involving robotic tasks applicable in both simulation and real world robotics. Additionally, we provide the tools to facilitate the creation of new environments featuring different robots and sensors.

Main features :
- [OpenAI Gym](https://gym.openai.com) interface for all the environments
- **simulated** and **real** robots interchangeability, which enables a seamless transfer from training in simulation to application on the real robot.
- built-in **distributed** capabilities, which enable the use of distributed algorithms and distributed hardware
- based only on **open source** software, which allows to develop applications on own hardware and without incurring in cloud services fees or software licensing costs
- integration of multiple commercially available **industrial robots**: MiR 100, Universal Robots (more to come)
- it has been successfully deployed to train a DRL algorithm to solve two different tasks in simulation that was able to solve the tasks on the real robots as well, without any further training in the real world

A [paper](https://arxiv.org/abs/2007.02753) describing robo-gym has been accepted for IROS 2020. A video showcasing the toolkit's
capabilities and additional info can be found on our [website](https://sites.google.com/view/robo-gym)

**NOTE**: We are continuously working to improve and expand robo-gym. If you are interested in reproducing the results obtained in the IROS 2020 paper please refer to v.0.1.0 for all the 3 repositories involved in the framework: [robo-gym](https://github.com/jr-robotics/robo-gym/tree/v0.1.0), [robo-gym-robot-servers](https://github.com/jr-robotics/robo-gym-robot-servers/tree/v0.1.0), [robo-gym-server-modules](https://github.com/jr-robotics/robo-gym-server-modules/tree/v0.1.0).

[See the News section](#news)

<!-- omit in toc -->
# Table of Contents

- [Basics](#basics)
- [Installation](#installation)
- [How to use](#how-to-use)
- [Environments](#environments)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [News](#news)


# Basics
[back to top](#robo-gym)

The ``robo-gym`` framework is composed of several building blocks.
Detailed information on them is given [here](docs/the_framework.md) and in the [paper](https://arxiv.org/abs/2007.02753).

![robo-gym framework](https://user-images.githubusercontent.com/36470989/79330117-4498dc80-7f19-11ea-9de4-bed4f6390f3a.jpg)

The framework can be subdivided in two main parts:

- The *Robot Server Side* (in green) is the one directly interacting with the real robot or
  simulating the robot. It is based on ROS, Gazebo and Python.
  It includes the robot simulation itself, the drivers to communicate with
  the real robot and additional required software tools.

- The *Environment Side* is the one providing the OpenAI Gym interface to the robot
  and implementing the different environments.

The *Robot Server Side* and the *Environment Side* can run on the same PC or on different PCs
connected via network.

# Installation
[back to top](#robo-gym)

<!-- omit in toc -->
## Environment Side
**Requirements:** Python >= 3.6

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

<!-- omit in toc -->
## Robot Server Side
**Requirements:** Ubuntu 20.04 (recommended) or 18.04.

The Robot Server Side can be installed on the same machine running the Environment Side
and/or on other multiple machines.

Install [robo-gym-robot-servers](https://github.com/jr-robotics/robo-gym-robot-servers)
following the instructions in the repository's README.

# How to use
[back to top](#robo-gym)

Each environment comes with a version to be run with a simulated version of the
robot and the scenario and version to be run with the real robot.
Simulated environments have a name ending with *Sim* whereas real robot environments
have a name ending with *Rob*.

<!-- omit in toc -->
## Simulated Environments

Before making a simulated environment it is necessary to start the Server Manager with: 

```sh
start-server-manager
```

The Server Manager takes care of starting and managing the correct simulation/s and Robot Server/s. 
Depending on the setup that you choose, the Server Manager could be running on the same machine on which you call  ``env.make()`` or on another machine connected via network.

The Server Manager is part of the [robo-gym-server-modules](https://github.com/jr-robotics/robo-gym-server-modules) package. A list of commands is available [here](https://github.com/jr-robotics/robo-gym-server-modules#how-to-use). 

To start an environment use:
```python
import gym, robo_gym

env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>')
env.reset()
```

The IP address of the machine on which the Server Manager is running has to
be passed as an argument to ``env.make``, if the Server Manager is running on the
same machine use ``ip='127.0.0.1'``.

To start a simulated environment with **GUI** use the optional *gui* argument:

```python
env = gym.make('EnvironmentNameSim-v0', ip='<server_manager_address>', gui=True)
```

<!-- omit in toc -->
### Additional commands for Simulated Environments

The Simulation wrapper provides some extra functionalities to the Simulated Environments.

- `env.restart_sim()` restart the simulation
- `env.kill_sim()` kill the simulation 

<!-- omit in toc -->
### Exception Handling Wrapper

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

<!-- omit in toc -->
## Real Robot Environments

When making a real robot environment the Robot Server needs to be started manually, see [here](https://github.com/jr-robotics/robo-gym-robot-servers#how-to-use) how to do that.  

Once the Real Robot Server is running, you can start the corresponding environment with: 

```python
import gym, robo_gym

env = gym.make('EnvironmentNameRob-v0', rs_address='<robot_server_address>')

env.reset()
```
The `<robot_server_address>` has to be formed as `IP:PORT`

# Environments 
[back to top](#robo-gym)

See [List of Environments](docs/environments.md).

For information on creating your own environments, see [Creating your own Environments](docs/creating_environments.md).

# Examples
[back to top](#robo-gym)

<!-- omit in toc -->
## Random Agent MiR100 Simulation Environment
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
    env.reset()
    while not done:
        # random step in the environment
        state, reward, done, info = env.step(env.action_space.sample())
```



Additional examples can be found [here](docs/examples)

# Testing 
[back to top](#robo-gym)

Start the Server Manager and attach to the session with: 

```sh
start-server-manager && attach-to-server-manager
```

<details>
<summary>Expected output</summary>
<p>

```sh
2021-XX-XX XX:XX:XX,XXX - serverManager - INFO - Server Manager started at 50100
```

For problems at this step see the [Testing](https://github.com/jr-robotics/robo-gym-server-modules#testing) section of [robo-gym-server-modules](https://github.com/jr-robotics/robo-gym-server-modules).
</p>
</details>

<br>

On the PC where you are running ``robo-gym`` associate the IP of the pc on which the Server Manager is running to the hostname `robot-servers` with:

```sh
sudo sh -c 'printf "127.0.0.1 robot-servers" >> /etc/hosts'
```

If you are running the Server Manager on a different PC replace *127.0.0.1* with the IP address of the machine. 

We are using [pytest](http://doc.pytest.org/) for tests. You can run a short selection of tests with:

```sh
pytest -m "not nightly"
```

or the full test suite with:

```sh
pytest
```

Once you are done run `kill-server-manager` to kill the Robot Server and the Server Manager.

# Troubleshooting 

If you encounter troubles running robo-gym please take a look at the [existing issues](https://github.com/jr-robotics/robo-gym/issues?q=is%3Aissue++), if you still cannot find solution to your problem please submit a [new issue](https://github.com/jr-robotics/robo-gym/issues/new/).

[Troubleshooting robo-gym-robot-servers](https://github.com/jr-robotics/robo-gym-robot-servers#troubleshooting)
[Troubleshooting robo-gym-server-modules](https://github.com/jr-robotics/robo-gym-server-modules#troubleshooting)

# Contributing
[back to top](#robo-gym)

New environments and new robots and sensors implementations are welcome!

More details and guides on how to contribute will be added soon!

If you have general questions or ideas that you would like to share please [start a new discussion](https://github.com/jr-robotics/robo-gym/discussions/new).

# Citation
[back to top](#robo-gym)

```
@article{lucchi2020robo,
  title={robo-gym--An Open Source Toolkit for Distributed Deep Reinforcement Learning on Real and Simulated Robots},
  author={Lucchi, Matteo and Zindler, Friedemann and M{\"u}hlbacher-Karrer, Stephan and Pichler, Horst},
  journal={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```
# News
[back to top](#robo-gym)

- 2021-05-19 (v1.0.0)
  + Added support for all the Universal Robots models: UR3, UR3e, UR5, UR5e, UR10, UR10e, UR16e
  + The Robot Server state can now be defined as a dictionary instead of a list to reduce errors caused by wrong indexing
  + Added support for Python 3.7, 3.8, 3.9 and stopped support for 3.5
  + Added Obstacle Avoidance environments  
  + Improved logging and debugging
  + Improved code quality and readability 

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
