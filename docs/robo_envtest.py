#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import math
import signal
import sys
import time

# noinspection PyUnresolvedReferences
import robo_gym

import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TimeLimit
from numpy.typing import NDArray

from utils.table import Table, write_csv
from utils.utils import flatten_to_dict

ROBOT_TYPE_MIR100 = "Mir100"
ROBOT_TYPE_UR = "UR"
ROBOT_TYPE_PANDA = "Panda"

terminating = False


def signal_handler(the_signal, frame):
    global terminating
    print("\nTermination requested")
    terminating = True
    # sys.exit(0)


def str_to_bool(str_val) -> bool:
    str_val = str_val.strip().lower()
    return str_val not in ["", "0", "false", "no", "n"]


def str_for_log(val: float) -> str:
    val = round(val, 9)
    return f"{val:10.9}"


def main():
    global terminating
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--action_mode",
        help="Action mode (abs_pos, delta_pos, abs_vel, delta_vel)",
        default="abs_pos",
    )
    parser.add_argument(
        "-c", "--csvlog", help="file to write CSV table log", default=""
    )
    parser.add_argument(
        "-e", "--env", help="RL environment", default="NoObstacleNavigationMir100Sim-v0"
    )
    parser.add_argument(
        "-et",
        "--episode_timesteps",
        help="max time steps within episode",
        default="300",
    )
    parser.add_argument("-g", "--gui", help="GUI enabled", default="true")
    parser.add_argument(
        "-gz", "--gazebo_gui", help="Gazebo GUI enabled", default="true"
    )
    parser.add_argument(
        "-i", "--ip", help="server manager ip address or host name", default="ROBL1003a"
    )
    parser.add_argument(
        "-p", "--policy", help="policy trained with rsl rl in Isaac Lab", default=""
    )
    parser.add_argument("-rs", "--rsaddress", help="robot server address", default="")
    parser.add_argument("-rv", "--rviz_gui", help="RViz enabled", default="true")
    parser.add_argument(
        "-t", "--timesteps", help="max time steps overall", default="1000"
    )

    parser.add_argument(
        "-u", "--urmodel", help="UR model (ur3, ur5, ur10, ur16)", default=""
    )

    args = parser.parse_args(sys.argv[1:])

    np.set_printoptions(suppress=True)

    target_machine_ip = args.ip
    env_name = str(args.env)
    gui = str_to_bool(args.gui)
    gazebo_gui = gui and str_to_bool(args.gazebo_gui)
    rviz_gui = gui and str_to_bool(args.rviz_gui)
    rs_address = args.rsaddress
    timesteps = int(args.timesteps)
    episode_timesteps = int(args.episode_timesteps)
    ur_model = args.urmodel
    policy = None
    if args.policy:
        policy = IsaacPolicyWrapper(policy_file_path=args.policy)

    unversioned_env_name = env_name.split("-")[0]

    # remove Sim/Rob
    is_real_robot = unversioned_env_name.endswith("Rob")
    env_class_name = unversioned_env_name[:-3]

    is_robot_type = {}
    for robot_type in [ROBOT_TYPE_MIR100, ROBOT_TYPE_UR, ROBOT_TYPE_PANDA]:
        is_robot_type[robot_type] = env_class_name.startswith(
            robot_type
        ) or env_class_name.endswith(robot_type)

    kwargs = {"gazebo_gui": gazebo_gui, "rviz_gui": rviz_gui, "rs_state_to_info": True}

    action_mode = "abs_pos"
    is_avoidance = env_class_name.find("Avoidance") > -1
    is_ee_pos = env_class_name.find("EndEffectorPositioning") > -1

    if is_robot_type[ROBOT_TYPE_UR]:
        kwargs["ur_model"] = ur_model
        kwargs["randomize_start"] = False
    if is_robot_type[ROBOT_TYPE_PANDA]:
        action_mode = args.action_mode
        # kwargs["action_mode"] = action_mode
        # kwargs["action_cycle_rate"] = 25

    # Where in the observations do the joint position start?
    joint_pos_obs_offset = 0
    if robot_type != ROBOT_TYPE_MIR100 and (is_ee_pos or is_avoidance):
        joint_pos_obs_offset += 3  # target polar coordinates

    if rs_address:
        env = gym.make(env_name, rs_address=rs_address, gui=gui, **kwargs)
    else:
        env = gym.make(env_name, ip=target_machine_ip, gui=gui, **kwargs)

    env = TimeLimit(env, episode_timesteps)

    time.sleep(1)
    time_count = 0
    period = episode_timesteps / 2

    direction = 1
    episode = 0
    all_done = False

    action_sample: NDArray = env.action_space.sample()
    zero_action = np.zeros_like(action_sample)
    action_length = len(action_sample)
    obs_sample: NDArray = env.observation_space.sample()

    log_table: Table | None = None
    if args.csvlog:
        columns = sorted(flatten_to_dict(obs_sample.tolist(), prefix="obs").keys())
        columns.extend(
            sorted(flatten_to_dict(action_sample.tolist(), prefix="action").keys())
        )
        log_table = Table(columns=columns)

    while not all_done:
        if time_count >= timesteps:
            break
        episode += 1
        episode_time_count = 0
        print("starting episode " + str(episode))
        observation, info = env.reset()
        done = False

        while time_count < timesteps and not (done or all_done):
            if policy:
                action = policy.compute_action(observation)
            else:
                # generate action with some very simple movement, depending on robot and some env details
                if not period:
                    param = 1
                else:
                    param = math.sin((episode_time_count / period) * math.tau)

                if is_robot_type[ROBOT_TYPE_UR]:
                    normalized_joint_positions = observation[
                        0 + joint_pos_obs_offset : action_length + joint_pos_obs_offset
                    ]
                    delta_action = np.copy(zero_action)
                    delta_action[0] = param * 0.02 * direction

                    action = normalized_joint_positions + delta_action
                    action = action.astype(dtype=np.float32)
                    if not env.action_space.contains(action):
                        direction = -direction
                        action = normalized_joint_positions  # no move this time
                        action = action.astype(dtype=np.float32)
                    if not env.action_space.contains(action):
                        raise Exception("Fix the action math")
                elif is_robot_type[ROBOT_TYPE_PANDA]:
                    if action_mode == "abs_pos":
                        normalized_joint_positions = observation[
                            0
                            + joint_pos_obs_offset : action_length
                            + joint_pos_obs_offset
                        ]
                        action = normalized_joint_positions
                        # assume that joint 0 has starting position 0, otherwise needs a different solution for the start
                        action[0] = param * 0.1
                        action = action.astype(dtype=np.float32)
                        if not env.action_space.contains(action):
                            raise Exception("Fix the action math")
                    elif action_mode == "delta_pos":
                        action = np.copy(zero_action)
                        action[0] = param * 0.05
                    else:
                        action = np.copy(zero_action)
                        action[0] = param
                elif is_robot_type[ROBOT_TYPE_MIR100]:
                    action = np.array([0.05, 1.0], dtype=np.float32)
                else:
                    if is_real_robot and rs_address:
                        print("Can't handle unknown real robots")
                        break
                    action = env.action_space.sample()
            if log_table:
                row = flatten_to_dict(observation.tolist(), prefix="obs")
                row.update(flatten_to_dict(action.tolist(), prefix="action"))
                log_table.add_row(row)

            observation, reward, terminated, truncated, info = env.step(action)
            episode_time_count += 1
            time_count += 1
            all_done = terminating or time_count >= timesteps
            done = terminated or truncated
            final_status = "unknown"

            if done:
                if "final_status" in info:
                    final_status = info["final_status"]
                    if final_status == "max_steps_exceeded":
                        terminated = False
                        truncated = True
                print(
                    "Episode {} finished.\nTime steps: {}\nTerminated: {}\nTruncated: {}\nFinal status: {}\nReward: {}\n".format(
                        episode,
                        episode_time_count,
                        terminated,
                        truncated,
                        final_status,
                        reward,
                    )
                )
                print_info(info)

            elif all_done:
                print(
                    "Shutting down - Episode {} not finished.\nTime steps: {}\nTerminated: {}\nTruncated: {}\nFinal status: {}\nReward: {}\n".format(
                        episode,
                        episode_time_count,
                        terminated,
                        truncated,
                        final_status,
                        reward,
                    )
                )
                print_info(info)

            if all_done:
                env.close()
                if log_table:
                    write_csv(log_table, args.csvlog, value_formatters=str_for_log)

    # redundant - killing simulation upon object cleanup anyway
    # if not is_real_robot:
    #    try:
    #        env.unwrapped.kill_sim()
    #    except:
    #        pass


def print_info(info):
    for key in sorted(info):
        print("- " + key + ": " + str(info[key]))


class IsaacPolicyWrapper:
    # based on https://github.com/louislelay/isaaclab_ur_reach_sim2real

    def __init__(self, policy_file_path: str):
        import torch

        with open(policy_file_path, "rb") as f:
            file = io.BytesIO(f.read())
        self.policy = torch.jit.load(file)

    def compute_action(self, obs: NDArray) -> NDArray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        import torch

        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action


if __name__ == "__main__":
    main()
    print("main ending normally")
