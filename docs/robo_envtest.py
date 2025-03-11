#!/usr/bin/env python
import math
import time
import numpy as np
import gymnasium as gym
import sys
import signal
import argparse
from gymnasium.wrappers import TimeLimit
import robo_gym
from envs.base.robogym_env import RoboGymEnv
from robo_gym.wrappers.exception_handling import ExceptionHandling

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
    parser.add_argument("-rs", "--rsaddress", help="robot server address", default="")
    parser.add_argument("-rv", "--rviz_gui", help="RViz enabled", default="true")
    parser.add_argument(
        "-t", "--timesteps", help="max time steps overall", default="1000"
    )

    parser.add_argument(
        "-u", "--urmodel", help="UR model (ur3, ur5, ur10, ur16)", default=""
    )

    # TODO logic for real/sim distinction and name structure

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
    is_avoidance = env_class_name.find("Avoidance")

    if is_robot_type[ROBOT_TYPE_UR]:
        kwargs["ur_model"] = ur_model
        kwargs["randomize_start"] = False
    if is_robot_type[ROBOT_TYPE_PANDA]:
        action_mode = args.action_mode
        kwargs["action_mode"] = action_mode
        kwargs["action_cycle_rate"] = 25

    # Where in the observations do the joint position start?
    joint_pos_obs_offset = 0
    if robot_type != ROBOT_TYPE_MIR100 and (
        env_class_name.find("EndEffectorPositioning") > -1 or is_avoidance > -1
    ):
        joint_pos_obs_offset += 3  # target polar coordinates

    if rs_address:
        env = gym.make(env_name, rs_address=rs_address, gui=gui, **kwargs)
    else:
        env = gym.make(env_name, ip=target_machine_ip, gui=gui, **kwargs)
        # env = ExceptionHandling(env)

    env = TimeLimit(env, episode_timesteps)

    observation, info = env.reset()
    time.sleep(1)
    time_count = 0
    period = episode_timesteps / 2

    direction = 1
    episode = 0
    all_done = False

    while not all_done:
        if time_count >= timesteps:
            break
        episode += 1
        episode_time_count = 0
        print("starting episode " + str(episode))
        observation, info = env.reset()
        done = False

        while time_count < timesteps and not (done or all_done):
            if not period:
                param = 1
            else:
                param = math.sin((episode_time_count / period) * math.tau)

            if is_robot_type[ROBOT_TYPE_UR]:
                normalized_joint_positions = observation[
                    0 + joint_pos_obs_offset : 5 + joint_pos_obs_offset
                ]
                delta_action = (
                    np.array([param * 0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    * direction
                )
                action = normalized_joint_positions + delta_action
                action = action.astype(dtype=np.float32)
                if not env.action_space.contains(action):
                    direction = -direction
                    action = normalized_joint_positions  # no move this time
                    action = action.astype(dtype=np.float32)
                if not env.action_space.contains(action):
                    raise Exception("Fix the action math")
            elif is_robot_type[ROBOT_TYPE_PANDA]:
                if args.action_mode == "abs_pos":
                    normalized_joint_positions = observation[
                        0 + joint_pos_obs_offset : 6 + joint_pos_obs_offset
                    ]
                    action = normalized_joint_positions
                    # assume that joint 0 has starting position 0, otherwise needs a different solution for the start
                    #
                    action[0] = param * 0.1
                    action = action.astype(dtype=np.float32)
                    if not env.action_space.contains(action):
                        raise Exception("Fix the action math")
                elif args.action_mode == "delta_pos":
                    action = np.array(
                        [param * 0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    )
                else:
                    action = np.array(
                        [param, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    )
            elif is_robot_type[ROBOT_TYPE_MIR100]:
                action = np.array([0.05, 1.0])
            else:
                if is_real_robot and rs_address:
                    print("Can't handle unknown real robots")
                    break
                action = env.action_space.sample()
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
                env.close()

    # redundant - killing simulation upon object cleanup anyway
    # if not is_real_robot:
    #    try:
    #        env.unwrapped.kill_sim()
    #    except:
    #        pass


if __name__ == "__main__":
    main()
    print("main ending normally")
