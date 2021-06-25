#!/usr/bin/env python3
import numpy as np
import gym
from typing import Tuple
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation


class ExampleEnv(gym.Env):
    """Example environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
  
    Attributes:

        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 300

    def __init__(self, rs_address=None, rs_state_to_info=True, **kwargs):
        self.elapsed_steps = 0

        self.rs_state_to_info = rs_state_to_info

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.rs_state = None
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")


    def _set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                            string_params = string_params, state_dict = rs_state)
        return state_msg

    def reset(self, position = None) -> np.array:
        """Environment reset.

        Args:
            position (list[2] or np.array[2]): [x,y] initial robot position.
           
        Returns:
            np.array: Environment state.

        """
        # Set Robot starting position
        if position:
            assert len(position)==2
        else:
            position = [0,0]

        self.elapsed_steps = 0

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)
        

        # Fill rs_state
        rs_state['pos_x'] = position[0]
        rs_state['pos_y'] = position[1]

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state)
        
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = self.client.get_state_msg().state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        # Convert the initial state from Robot Server format to environment format
        state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            raise InvalidStateError()

        self.rs_state = rs_state

        return state

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            
        
        return 0, done, info
   

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)
            
        self.elapsed_steps += 1

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            raise InvalidActionError()


        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(action.tolist()).state_dict
        self._check_rs_state_keys(rs_state)

        # Convert the state from Robot Server format to environment format
        state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            raise InvalidStateError()

        self.rs_state = rs_state

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self.reward(rs_state=rs_state, action=action)
        if self.rs_state_to_info: info['rs_state'] = self.rs_state

        return state, reward, done, info

    def get_rs_state(self):
        return self.rs_state

    def render():
        pass
    
    def get_robot_server_composition(self) -> list:
        rs_state_keys = [
            'pos_x',
            'pos_y',
            'lin_vel',
            'ang_vel',
        ]
        return rs_state_keys


    def _get_robot_server_state_len(self) -> int:
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.
        """
        return len(self.get_robot_server_composition())

    def _check_rs_state_keys(self, rs_state) -> None:
        keys = self.get_robot_server_composition()
        if not len(keys) == len(rs_state.keys()):
            raise InvalidStateError("Robot Server state keys to not match. Different lengths.")
        
        for key in keys:
            if key not in rs_state.keys():
                raise InvalidStateError("Robot Server state keys to not match")


    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """

        pos_x = rs_state['pos_x']
        pos_y = rs_state['pos_y']
        lin_vel = rs_state['lin_vel']
        ang_vel = rs_state['ang_vel']

        # Compose environment state
        state = np.array([pos_x, pos_y, lin_vel, ang_vel])

        return state


    def _get_observation_space(self) -> gym.spaces.Box:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
 
        # Definition of environment observation_space
        max_obs = np.array([np.inf] * 4)
        min_obs = -np.array([np.inf] * 4)

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    
    def _get_action_space(self)-> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """

        return gym.spaces.Box(low=np.full(2, -1.0), high=np.full(2, 1.0), dtype=np.float32)


class ExampleEnvSim(ExampleEnv, Simulation):
    cmd = "roslaunch example_robot_server robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ExampleEnv.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ExampleEnvRob(ExampleEnv):
    real_robot = True

# roslaunch example_robot_server robot_server.launch  gui:=true real_robot:=true
