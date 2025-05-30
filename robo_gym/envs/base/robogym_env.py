from __future__ import annotations

import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.core import ObsType, ActType
from numpy.typing import NDArray
from types import UnionType
from typing import Any, SupportsFloat, Tuple, Callable

import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.utils.exceptions import RobotServerError
from robo_gym.utils.managed_rs_client import ManagedClient
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2


class RoboGymEnv(gym.Env):
    """
    Base class for robo-gym environments, linked to a robot server that wraps a simulated or real robot.
    """

    KW_RS_ADDRESS = "rs_address"
    KW_SERVER_MANAGER_HOST = "ip"
    KW_SERVER_MANAGER_PORT = "server_manager_port"
    KW_REAL_ROBOT = "real_robot"
    KW_ACTION_RATE = "action_rate"
    KW_GUI_FLAG = "gui"
    KW_GAZEBO_GUI_FLAG = "gazebo_gui"
    KW_RVIZ_GUI_FLAG = "rviz_gui"
    KW_ROBOT_MODEL_KEY = "robot_model"
    KW_ROBOT_MODEL_OBJECT = "robot_model_object"
    KW_RS_STATE_TO_INFO = "rs_state_to_info"

    KW_ACTION_NODE = "action_node"
    KW_REWARD_NODE = "reward_node"
    KW_OBSERVATION_NODES = "observation_nodes"

    INFO_KW_RS_STATE = "rs_state"
    INFO_KW_FINAL_STATUS = "final_status"
    FINAL_STATUS_SUCCESS = "success"
    FINAL_STATUS_COLLISION = "collision"

    # max_steps_exceeded should be replaced by proper truncated mechanism
    FINAL_STATUS_MAX_STEPS_EXCEEDED = "max_steps_exceeded"

    def __init__(self, **kwargs):
        self._config = kwargs

        # client will be obtained lazily
        # 3 cases:
        # - new sim RS via server manager
        # - existing real RS
        # - existing sim RS (new case!)
        self._client: rs_client.Client | None = None

        self._action_node: ActionNode = kwargs.get(self.KW_ACTION_NODE)
        self._reward_node: RewardNode = kwargs.get(self.KW_REWARD_NODE)
        self._observation_nodes: list[ObservationNode] = [
            node for node in kwargs.get(self.KW_OBSERVATION_NODES)
        ]

        self._setup_nodes()

        self.action_space: gym.spaces.Box = self._get_action_space_no_cache()
        self.observation_space = self._get_observation_space_no_cache()

        self._elapsed_steps = 0
        self._episodes_count = 0

        self._elapsed_steps = 0
        self._last_env_action = np.zeros_like(self.action_space.high)
        self._last_rs_state_array: NDArray = np.array([], dtype=np.float32)
        self._last_rs_state_dict: dict[str, float] = {}

    def _setup_nodes(self):
        """
        Calls the setup method for each environment node
        Returns: nothing
        """
        self._action_node.setup(self, **self.get_action_node_setup_kwargs())
        self._reward_node.setup(self, **self.get_reward_node_setup_kwargs())
        for index in range(len(self._observation_nodes)):
            self._observation_nodes[index].setup(
                self, **self.get_obs_node_setup_kwargs(index)
            )

    def _reset_episode_fields(self):
        """
        Resets episode-related fields
        Returns: nothing

        """
        self._elapsed_steps = 0
        self._last_env_action = np.zeros_like(self.action_space.high)
        self._last_rs_state_array = np.array([], dtype=np.float32)
        self._last_rs_state_dict = {}

    def _get_action_space_no_cache(self) -> gym.spaces.Box:
        """
        Gets the action space returned by the action node.
        Returns: the action space returned by the action node

        """
        return self._action_node.get_action_space()

    def _get_observation_space_no_cache(self) -> gym.spaces.Box:
        """
        Gets the observation space returned by the observation nodes.
        Returns: the observation space obtained from concatenating the box spaces returned by the observation nodes

        """
        space_parts = [
            node.get_observation_space_part() for node in self._observation_nodes
        ]
        highs = [space_part.high for space_part in space_parts]
        lows = [space_part.low for space_part in space_parts]
        all_highs = np.concatenate(highs).astype(np.float32)
        all_lows = np.concatenate(lows).astype(np.float32)
        result = gym.spaces.Box(low=all_lows, high=all_highs, dtype=np.float32)
        return result

    def get_action_node_setup_kwargs(self):
        """
        Get arguments for calling setup on the action node. Defaults to own config. Override to provide different arguments.
        Returns: own config dict

        """
        return self._config

    def get_action_node_dyn_kwargs(self):
        return self._config

    def get_reward_node_setup_kwargs(self):
        return self._config

    def get_reward_node_dyn_kwargs(self):
        return self._config

    def get_obs_node_setup_kwargs(self, index: int):
        return self._config

    def get_obs_node_dyn_kwargs(self, index: int):
        return self._config

    def get_all_nodes(self) -> list[EnvNode]:
        result = [self._action_node, self._reward_node]
        result.extend(self._observation_nodes)
        return result

    def get_all_nodes_sorted(self) -> list[EnvNode]:
        result = [self._action_node, self._reward_node]

        def my_sort_key(obs_node: ObservationNode):
            return obs_node.get_reset_state_part_order()

        result.extend(sorted(self._observation_nodes, key=my_sort_key))
        return result

    def create_main_observation_node(self, node_index: int = 0, **kwargs):
        return ObservationNode(**self.get_obs_node_setup_kwargs(node_index))

    @property
    def client(self) -> rs_client.Client | None:
        if self._client is None:
            if self.rs_address:
                self._client = rs_client.Client(self.rs_address)
            elif self.server_manager_host:
                if self.real_robot:
                    raise Exception(
                        "Config determines to use real robot with server manager - cannot work!"
                    )
                self._client = ManagedClient(
                    server_manager_host=self.server_manager_host,
                    server_manager_port=self.server_manager_port,
                    launch_cmd=self.get_launch_cmd(),
                    gui=self.get_gui_flag(),
                )
            else:
                raise Exception(
                    "Config determines neither robot server nor server manager - cannot work!"
                )
        return self._client

    @property
    def rs_address(self) -> str:
        return self._config.get(self.KW_RS_ADDRESS)

    @property
    def server_manager_host(self) -> str:
        return self._config.get(self.KW_SERVER_MANAGER_HOST)

    @property
    def server_manager_port(self) -> int:
        return self._config.get(self.KW_SERVER_MANAGER_PORT, 50100)

    @property
    def real_robot(self) -> bool:
        return self._config.get(self.KW_REAL_ROBOT, False)

    @property
    def action_rate(self) -> float:
        return self._config.get(self.KW_ACTION_RATE)

    def get_launch_cmd(self) -> str:
        # TODO prepare to have this as an object first, stringify when needed
        # TODO derive command from components
        return 'rostopic pub my_topic std_msgs/String "you should have changed the launch command" -r 1'

    def get_gui_flag(self) -> bool:
        return self._config.get(self.KW_GUI_FLAG, False)

    def env_action_to_rs_action(self, env_action) -> NDArray:
        return self._action_node.env_action_to_rs_action(
            env_action, **self.get_action_node_dyn_kwargs()
        )

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # might want to add some checks here, but trivial ones may be done by wrappers already

        env_action = np.array(action)

        self._last_env_action = env_action
        self._elapsed_steps += 1

        rs_action = self.env_action_to_rs_action(env_action)

        rs_state: robot_server_pb2.State = self.client.send_action_get_state(rs_action)

        self._last_rs_state_array = np.array(rs_state.state)
        self._last_rs_state_dict = dict(rs_state.state_dict)

        env_observation = self.rs_state_to_env_obs(
            self._last_rs_state_array, self._last_rs_state_dict
        )
        reward, terminated, info = self._reward_node.get_reward(
            self._last_rs_state_array, self._last_rs_state_dict, env_action
        )

        truncated = False
        return env_observation, reward, terminated, truncated, info

    def rs_state_to_env_obs(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float]
    ) -> NDArray:
        obs_parts = []
        for index in range(len(self._observation_nodes)):
            obs_parts.append(
                self._observation_nodes[index].rs_state_to_observation_part(
                    rs_state_array, rs_state_dict, **self.get_obs_node_dyn_kwargs(index)
                )
            )
        result = np.concatenate(obs_parts)
        return result

    def _get_initial_state_msg(self) -> robot_server_pb2.State:
        # MiR: only the state array is used; the server takes it apart based on hardcoded positions
        # UR: param dicts and state dict are used; server would support using state array instead of state dict

        nodes = self.get_all_nodes_sorted()

        rs_state_string_params: dict[str, str] = {}
        rs_state_float_params: dict[str, float] = {}
        rs_state_dict: dict[str, float] = {}
        rs_state_array = np.array([], np.float32)
        for node in nodes:
            node_string_params = node.get_reset_state_part_str()
            node_float_params = node.get_reset_state_part_float()
            node_state_dict = node.get_reset_state_part_state_dict()
            node_state_array = node.get_reset_state_part_state_array_values()

            rs_state_string_params.update(node_string_params)
            rs_state_float_params.update(node_float_params)
            rs_state_dict.update(node_state_dict)
            rs_state_array = np.concatenate((rs_state_array, node_state_array))

        state_msg = robot_server_pb2.State(
            state=rs_state_array,
            float_params=rs_state_float_params,
            string_params=rs_state_string_params,
            state_dict=rs_state_dict,
        )
        return state_msg

    def _set_initial_server_state(self):
        state_msg = self._get_initial_state_msg()

        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

    def get_default_info(self) -> dict:
        """
        base values for the info dict
        Returns:

        """
        info = {}
        if self._config.get(self.KW_RS_STATE_TO_INFO, False):
            info[self.INFO_KW_RS_STATE] = self._last_rs_state_dict
        return info

    def _get_reset_info(self) -> dict:
        """
        the info dict to be returned by reset
        Returns: result of get_default_info

        """
        return self.get_default_info()

    def _prepare_state(self):
        """
        Called after setting new seed and before setting new state on robot server.
        Subclass should override this to perform any episode preparation depending on RNG.
        """
        pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed, options=options)
        # consider RNG (self.np_random) as ready for the new episode

        # TODO integrate arg check as introduced for Panda

        if self._episodes_count > 0:
            # if _episode_count == 0, this would be redundant
            self._setup_nodes()
            self._reset_episode_fields()
        self._episodes_count += 1

        self._prepare_state()

        # TODO reconsider if any methods called below here should be set up to receive (more) args in subclasses
        self._set_initial_server_state()
        rs_state = self.client.get_state_msg()

        self._last_rs_state_array = np.array(rs_state.state)
        self._last_rs_state_dict = dict(rs_state.state_dict)

        env_observation = self.rs_state_to_env_obs(
            self._last_rs_state_array, self._last_rs_state_dict
        )
        reset_info = self._get_reset_info()
        return env_observation, reset_info

    def close(self):
        client = self.client
        if isinstance(client, ManagedClient):
            client.kill()
        super().close()

    def get_config(self, key=None):
        if key == None:
            return self._config
        if self._config == None:
            return None
        return self._config[key]

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @staticmethod
    def assure_instance_of_type_in_list(
        kwargs: dict[str, Any],
        key: str,
        the_type: type | UnionType,
        insert_as_head: bool,
        initializer: Callable,
        initializer_args: dict[str, Any],
    ) -> object:
        the_list = kwargs.get(key)
        if the_list is None:
            the_list = []
        elif not isinstance(the_list, list):
            raise Exception("Expected list in kw args at key '" + key + "'")

        for existing_element in the_list:
            if isinstance(existing_element, the_type):
                return existing_element

        the_instance = initializer(initializer_args)
        if insert_as_head:
            the_list.insert(0, the_instance)
        else:
            the_list.append(the_instance)
        kwargs[key] = the_list
        return the_instance

    @property
    def last_env_action(self):
        return self._last_env_action


class EnvNode(ABC):

    def __init__(self, **kwargs):
        self._config = kwargs
        self.env: RoboGymEnv | None = None

    def setup(self, env: RoboGymEnv, **kwargs):
        # TODO ugly coupling, try to eliminate
        self.env = env
        self._config.update(kwargs)

    def get_reset_state_part_float(self) -> dict[str, float]:
        return {}

    def get_reset_state_part_str(self) -> dict[str, str]:
        return {}

    def get_reset_state_part_state_dict(self) -> dict[str, float]:
        return {}

    def get_reset_state_part_state_array_values(self) -> NDArray:
        return np.array([], dtype=np.float32)

    def get_reset_state_part_order(self) -> int:
        return 0


class ActionNode(EnvNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, env: RoboGymEnv, **kwargs):
        super().setup(env, **kwargs)

    @abstractmethod
    def get_action_space(self) -> gym.spaces.Box:
        pass

    @abstractmethod
    def env_action_to_rs_action(self, env_action: NDArray, **kwargs) -> NDArray:
        pass


class ObservationNode(EnvNode):

    @abstractmethod
    def get_observation_space_part(self) -> gym.spaces.Box:
        pass

    @abstractmethod
    def rs_state_to_observation_part(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float], **kwargs
    ) -> NDArray:
        pass


class RewardNode(EnvNode):

    KW_MAX_EPISODE_STEPS = "max_episode_steps"

    def get_reward(
        self,
        rs_state_array: NDArray,
        rs_state_dict: dict[str, float],
        env_action: NDArray,
        **kwargs,
    ) -> Tuple[float, bool, dict]:

        info = self.env.get_default_info()
        return 0.0, False, info

    @property
    def max_episode_steps(self) -> int | None:
        return self._config.get(RewardNode.KW_MAX_EPISODE_STEPS)


class LastActionObservationNode(ObservationNode):

    def get_observation_space_part(self) -> gym.spaces.Box:
        return self.env.action_space

    def rs_state_to_observation_part(
        self, rs_state_array: NDArray, rs_state_dict: dict[str, float], **kwargs
    ) -> NDArray:
        return self.env.last_env_action
