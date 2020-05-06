# The robo-gym framework

The following information is extracted from: *M. Lucchi, F. Zindler, S. Mühlbacher-Karrer, H. Pichler, "robo-gym - An Open Source Toolkit for Distributed Deep Reinforcement Learning on Real and Simulated Robots"* Submitted to IROS 2020. Under review. 

![alt text](https://user-images.githubusercontent.com/36470989/79330117-4498dc80-7f19-11ea-9de4-bed4f6390f3a.jpg "The robo-gym framework")

## The Components

### Real or Simulated Robot

This component includes the robot itself, the sensors, and the scene surrounding the robot.

The interface to the robots and the sensors is implemented in ROS for both real and simulated hardware. The interface of the simulated robots is generally corresponding to the one of the real robots augmented with features that would be impractical or too expensive to match in the real world. An example is virtual collision sensors that detect any kind of collision of a robot link.
The simulated and the real robot must use the same controllers.

The simulated scenes are generated in Gazebo and are described using the SDFormat (SDF), an XML format. These can be created and manipulated in multiple ways: online, via API or GUI, and offline, by editing the SDF files.

### Command Handler

Within the Markov Decision Process (MDP) framework, it is assumed that interactions between agent and environment take place at each of a series of discrete time steps.
In a real-world system, however, time passes in a continuous manner.
It is therefore necessary to make some adjustments to the real world system so that its behavior gets closer to the one defined by the MDP framework.
The Command Handler (CH) implements these aspects.

The CH uses a queue with capacity for one command message.
When the component receives a command message it tries to fill the queue with it.
New elements get ignored until the element in the queue gets consumed.   
The CH continuously publishes command messages to the robot at the frequency required by its controller.
If, at the moment of publishing, the queue is full, the CH retrieves the command, publishes it to the robot for the selected number of times and after that it empties the queue.
In the opposite case, the handler publishes the command corresponding to an interruption of the movement execution. This corresponds to either zero velocities for mobile robots or preemption of the trajectory execution for robot arms.

The framework's CH supports the standard *diff_drive_controller* and *joint_trajectory_controller* from  ROS controllers.
This covers the most common types of robots; nevertheless, this component can be easily implemented for any other ROS controller.

### Robot Server

It exposes a gRPC server that allows to interact with the robot through the integrated ROS bridge.

The first function of the server is to store updated information regarding the state of the robot, that can be queried at any time via a gRPC service call.
The robot and the sensors constantly publish information via ROS.
The ROS Bridge collects the information from the different sources and stores it in a buffer as an array of values.
The robot and the sensors update their information with different frequencies.
The buffer is managed with a threading mechanism to ensure that the data delivered to the client is consistent and containing the latest information.

The second function is to set the robot and the scene to a desired state. For instance, the user might want to set the joint positions of a robotic arm to a specific value when resetting the environment.

Lastly, it provides a service to publish commands to the CH.

### Environment

This is the top element of the framework, which provides the standard OpenAI Gym interface to the agent.

The main function of the Environment component is to define the composition of the state, the initial conditions of an episode and the reward function.
In addition, it includes a gRPC stub which connects to the Robot Server to send actions, and to set or get the state of the robot and the scene.

According to the framework provided by the Gym, environments are organized in classes, each constructed from a common base one. In addition, robo-gym extends this setup with a different wrapper for either a real or a simulated robot. These wrappers differentiate regarding the constructor that is being called.
In the case of the simulated robot environment, the argument for the IP address refers to the Server Manager, whereas in the case of the real robot environment it refers to the IP address of the Robot Server.
The Server Manager for simulated robots provides the address of the  Robot Server to which the Environment gRPC stub is then connected.
On the other hand, in the case of the real robot environment, extra attention for the network configuration is needed to guarantee communication with the hardware. Furthermore, environment failures and eventual emergency stops must be inspected by an human operator.
As a consequence, the Server Manager is currently not employed when using real robots and the Environment gRPC stub is connected directly to the Robot Server, which is started manually.

### Server Manager

It is the orchestrator of the Robot Servers, it exposes gRPC services to spawn, kill, and check Robot Servers.
When used with simulated robots it handles the robot simulations as well.

Each cluster of Robot Server, CH and real or simulated robot runs on an isolated ROS network.
To achieve this, the Server Manager launches each cluster in an isolated shell environment handled with the help of tmux.

This component implements error handling features to automatically restart the Robot Server and the robot simulation in case of:
- an error in the connection to the Robot Server
- an exceeded deadline when calling a Robot Server service
- a non responding simulation
- data received from simulation out of defined bounds
- a manual request of simulation restart
