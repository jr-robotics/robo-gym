import gymnasium as gym
from ray.rllib.policy.policy import Policy
import robo_gym
import ray
import sys
import numpy as np
from robo_gym.envs.simulation_wrapper import Simulation
import torch
import torch.nn
from ray.rllib.core.rl_module import RLModule

if __name__ == "__main__":

    policy_cp_path = sys.argv[1]

    policy = Policy.from_checkpoint(policy_cp_path)
    
    env = gym.make("NoObstacleNavigationMir100Sim-v0", ip="127.0.0.1", gui=True)
    obs, _ = env.reset()
    
    num_episodes = 0
    episode_reward = 0.0

    count_overall = 0
    count_clipped = 0

    while num_episodes < 5:
        torch_obs_batch = torch.from_numpy(np.array([obs]))
        a = policy.compute_single_action(torch_obs_batch)
        a_np = np.array(a[0], dtype=np.float32)
        
        clipped_a_np = a_np.clip(-1, 1)

        if not np.array_equal(clipped_a_np, a_np):
            count_clipped += 1

        count_overall += 1
        
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(clipped_a_np)
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, _ = env.reset()
            env.render()
            num_episodes += 1
            episode_reward = 0.0

    env.close()

    ray.shutdown()
    print(f"clipped {count_clipped} of {count_overall} actions")