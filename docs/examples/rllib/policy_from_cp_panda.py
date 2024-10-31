import gymnasium as gym
from ray.rllib.policy.policy import Policy
import robo_gym
import ray
import sys
import time
import numpy as np
import torch
import torch.nn
from ray.rllib.core.rl_module import RLModule

if __name__ == "__main__":

    policy_cp_path = sys.argv[1]
    action_mode = "abs_pos"
    if len(sys.argv) > 1:
        action_mode = sys.argv[2]

    policy = Policy.from_checkpoint(policy_cp_path)
    
    env = gym.make("EndEffectorPositioningPandaSim-v0", ip="127.0.0.1", gui=True, action_mode=action_mode, rs_state_to_info=False)
    obs, _ = env.reset()
    
    num_episodes = 0
    episode_reward = 0.0

    count_overall = 0
    count_clipped = 0

    step_duration_overall = 0.0

    while num_episodes < 5:
        step_start_time = time.time()
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

        step_end_time = time.time()
        current_step_duration = (step_end_time - step_start_time)
        # print(f"step duration: {current_step_duration}")
        step_duration_overall += current_step_duration        
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, _ = env.reset()
            num_episodes += 1
            episode_reward = 0.0

    env.close()

    ray.shutdown()
    print(f"clipped {count_clipped} of {count_overall} actions")
    if count_overall:
        print(f"step duration: {step_duration_overall/count_overall} avg, {step_duration_overall} total")    