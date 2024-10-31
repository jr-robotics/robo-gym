"""
Example showing how you can use your trained policy for inference
(computing actions) in an environment.

Includes options for LSTM-based models (--use-lstm), attention-net models
(--use-attention), and plain (non-recurrent) models.
"""
import argparse
import gymnasium as gym
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import get_trainable_cls, register_env

import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from robo_gym.envs.panda.panda_ee_positioning import EndEffectorPositioningPandaSim

def env_creator_panda(config):
    return ExceptionHandling(EndEffectorPositioningPandaSim(**config))


register_env(
    name="EndEffectorPositioningPandaSim-v0",
    env_creator=env_creator_panda,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--stop-iters",
    type=int,
    default=2000,
    help="Number of iterations to train before we do inference.",
)
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=40000,
    help="Number of timesteps to train before we do inference.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=150.0,
    help="Reward at which we stop training before we do inference.",
)
parser.add_argument(
    "--explore-during-inference",
    action="store_true",
    help="Whether the trained policy should use exploration during action "
    "inference.",
)
parser.add_argument(
    "--num-episodes-during-inference",
    type=int,
    default=2,
    help="Number of episodes to do inference over after training.",
)
parser.add_argument(
    "--action-mode",
    choices=["abs_pos", "delta_pos", "abs_vel", "delta_vel"],
    default="abs_pos",
    help="The action mode for controlling the robot.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    default_config = get_trainable_cls(args.run).get_default_config();
    config = (
        default_config
        .environment("EndEffectorPositioningPandaSim-v0", env_config={"ip": "127.0.0.1", "gui": False, "action_mode": args.action_mode, "rs_state_to_info": False},)
        # Run with tracing enabled for tf2?
        .framework(args.framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    print("Training policy until desired reward/timesteps/iterations. ...")
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1, checkpoint_at_end=True
            ),
        ),
    )
    results = tuner.fit()

    print("Training completed. Restoring new Algorithm for action inference.")
    # Get the last checkpoint from the above training run.
    checkpoint = results.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.
    algo = Algorithm.from_checkpoint(checkpoint)

    # Create the env to do inference in.
    env = gym.make("EndEffectorPositioningPandaSim-v0", ip="127.0.0.1", gui=True, action_mode=args.action_mode, rs_state_to_info=False)
    obs, info = env.reset()
    env.render()

    num_episodes = 0
    episode_reward = 0.0

    while num_episodes < args.num_episodes_during_inference:
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=args.explore_during_inference,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(a)
        env.render()
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = env.reset()
            env.render()
            num_episodes += 1
            episode_reward = 0.0

    algo.stop()
    env.close()
    ray.shutdown()
