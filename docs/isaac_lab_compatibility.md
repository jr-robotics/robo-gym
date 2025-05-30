# Isaac Lab Compatibility

The environments of robo-gym include a few that are intended to provide compatibility with agents trained using [Isaac Lab](https://developer.nvidia.com/isaac/lab). The main purpose of this compatibility is to offer robo-gym as a way of transferring the trained policies from Isaac Lab to real robots, with robo-gym simulations as an intermediate step.

An overview of the robo-gym environments implemented for this feature is included in [Modular Environments](modular_environments.md).

## Setup

### robo-gym

#### Agent side

Install robo-gym version >=2.1.0 and its prerequisites, including robo-gym server modules version 0.3.0.1.

In addition, install Torch:

`pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`

#### Server side

Install robo-gym-robot-servers >=2.1.1 and its prerequisites, including robo-gym server modules version 0.3.0.1 and prerequisites for the robots you want to use in your ROS noetic Python environment.

### Isaac Lab

Isaac Lab is not required to run our compatibility environments.

If you already have a policy that you want to use, you don't need to run Isaac Lab.

It is recommended to use a different Python environment for Isaac Lab and robo-gym (both sides), as their dependencies may be incompatible.

Install Isaac Lab as by their [instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation). 

Tested with policies from this setup:
* Isaac Sim: 4.5.0, installed via pip
* Isaac Lab: 2.1.0/commit 5af5f388 (main as of 29 April 2025)
* Python: 3.10.11
* OS: Windows 10 Enterprise Build 19045

## Training

Use the rsl_rl train script to perform training either on the Franka Reach or UR10 Reach tasks. The commands are executed from the Isaac Lab folder with the corresponding Python environment active.

See also the [overview of environments in Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html).

If not running on Windows, replace `isaaclab.bat` by the corresponding script file name for your system.

### Franka Emika Panda

Default training:

`isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Franka-v0 --headless`

Play the environment with the trained policy from the last run (needs to be started at least once to obtain exported policy file):

`isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Franka-v0`

### UR10 

Default training:

`isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-UR10-v0 --headless`

Play the environment with the trained policy from the last run (needs to be started at least once to obtain exported policy file):

`isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-UR10-v0`

## Running

### Server side

#### Simulation

In a commandline where you have your robo-gym server-side ROS workspace ready:

`start-server-manager && attach-to-server-manager`

#### Real robot

The procedure would be the same as for other environments for the corresponding robot. Due to limited progress in our own tests with real robots, we do not hand out instructions for this case right now.

### Agent side

We use our [env test script](robo_envtest.py) for convenient testing and to show that the Isaac Lab Compatibility environments are to be used in the same way as our other environments.

Arguments that need adaptation for your setup:
* `-i`: server manager host
* `-p`: Path to the exported policy

The commands are executed from this folder with the robo-gym Python environment active.

#### UR10

Command to run the [env test script](robo_envtest.py):

`python robo-envtest.py -i localhost -g 1 -gz 1 -rv 0 -e IsaacReachURSim-v0 -u ur10 -et 500 -t 5000 -p D:\\PATH\\TO\\IsaacLab\\logs\\rsl_rl\\reach_ur10\\2025-04-30_09-36-28\\exported\\policy.pt`

Your console output should give you information about the progress while a few episodes are executed. During each episode, you should see the UR10 in Gazebo reach the target to a degree comparable to what you see when playing in Isaac Lab. Mind that the orientation of the target pose cannot be seen in Gazebo. The console output of the test script should give you a success indication at the end of each episode, with the distance in translation ("dist_coord") in m and the distance in rotation ("dist_rot") in rad. As by the default configuration of the environment, the target is located within a certain cuboid and rotated randomly around the vertical axis.

The environment identifier usable in the `gym.make` command is `IsaacReachURSim-v0`.

#### Panda

Command to run the [env test script](robo_envtest.py):

`python robo-envtest.py -i localhost -g 1 -gz 1 -rv 0 -e IsaacReachPandaSim-v0 -et 500 -t 5000 -p D:\\PATH\\TO\\IsaacLab\\logs\\rsl_rl\\franka_reach\\2025-05-16_17-53-15\\exported\\policy.pt`

Note that, while similar results should be reachable for the Panda as for the UR10 (granted that significantly longer training time may be required), we have not yet achieved satisfactory performance with controlling the Panda robot via this approach.

The environment identifier usable in the `gym.make` command is `IsaacReachPandaSim-v0`.
