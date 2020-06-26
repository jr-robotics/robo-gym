# Creating Environments

One of the benefits of the modular architecture of robo-gym is that it is fairly simple to add new environments to it.

All the environments are stored under the [envs](robo_gym/envs) directory, in here we organized the different environments in different folders for each robot.

All the environments are based on the `gym.env` class, you can base your environment class directly on `gym.env` or start from one of the classes already included in robo-gym and expand them.

Let's take as an example the UR 10 environments, [UR10Env](https://github.com/jr-robotics/robo-gym/blob/86be4ff841df099c1513650b5d5f750cdd348ca0/robo_gym/envs/ur10/ur10.py#L13) is the base class used for all the environments that use the UR 10 robot, [EndEffectorPositioningUR10](https://github.com/jr-robotics/robo-gym/blob/86be4ff841df099c1513650b5d5f750cdd348ca0/robo_gym/envs/ur10/ur10.py#L267) inherits from [UR10Env](https://github.com/jr-robotics/robo-gym/blob/86be4ff841df099c1513650b5d5f750cdd348ca0/robo_gym/envs/ur10/ur10.py#L13) and replaces the `reward()` method to model a task to teach the robot to reach and End Effector Pose. Each environment will have a python class associated to it.

Once you created the class for your environment it has to be added to the registration in order to make it available. Following what it has already been done for the existing environments, add the import of your classes to [robo_gym/envs/\_\_init\_\_.py](robo_gym/envs/__init__.py) and the registration of the environment to [robo_gym/\_\_init\_\_.py](robo_gym/__init__.py).


## Integrating new robot and sensors

Integrating new robots and sensors is possible but requires knowledge of ROS and Gazebo, if you are interested in that we would be happy to support you with that, please reach out!
