import gym
import numpy as np
from typing import Tuple


class ReachAndAvoidURTrainingCurriculum(gym.Wrapper):
    def __init__(self, env, print_reward=False):
        super().__init__(env)
        self.env = env
        self.max_episode_steps = 500

        # use counter as metric for level up
        self.episode_counter = 0

        self.reward_composition = {}
        self.print_reward = print_reward
        


    def reset(self, **kwargs):
        if self.episode_counter % 5 == 0:
            state = self.env.reset(randomize_start=True)
        else:
            state = self.env.reset(continue_on_success=True)

        self.reward_composition = { 'goal_reached_weight': 0,
                                    'collision_weight': 0,
                                    'distance_weight': 0,
                                    'smoothness_weight': 0,
                                    'action_magnitude_weight': 0,
                                    'velocity_magnitude_weight': 0,
                                    'obstacle_distance': 0,
                                    'obstacle_collision': 0}

        return state

    def step(self, action):
        self.previous_action = self.env.previous_action
        next_state, _, _, _ = self.env.step(action)

        action = self.env.add_fixed_joints(action)
        reward, done, info = self.reward(rs_state=self.env.rs_state, action=action)

        if done:
            self.episode_counter += 1
        
        if done and self.print_reward:
            print(f'Episode counter: {self.episode_counter}   Current level: {self.get_level()}')
            print(self.reward_composition)

        return next_state, reward, done, info

    def get_level(self):
        level_thresholds = [75, 250, 500, 1000, 1500, 2500]
        
        for i in range(len(level_thresholds)):
            if self.episode_counter < level_thresholds[i]:
                return i+1

        return len(level_thresholds) + 1

    def get_weights(self, level):
        # weights

        # reward for reaching the goal position
        g_w = 2
        # reward for collision (ground, table or self)
        c_w = -1
        # punishment according to the distance to the goal
        d_w = -0.005
        # obstacle distance weight
        o_d_w = -0.001
        # obstacle collision weight
        o_c_w = -0.1
        # punishment delta in two consecutive actions
        s_w = -0.0002
        # punishment for acting in general
        a_w = -0.0001
        # punishment for deltas in velocity
        v_w = -0.0002

        if level == 1:
            # learn how to reach a target position
            self.min_distance = 0.15
            d_w = d_w * 1

            # do not consider obstacle
            o_d_w = o_d_w * 0
            o_c_w = o_c_w * 0
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.0

            # no smoothness
            s_w = s_w * 0
            a_w = a_w * 0
            v_w = v_w * 0
        if level == 2:
            # learn how to reach a target position
            self.min_distance = 0.15
            d_w = d_w * 0

            # do not consider obstacle
            o_d_w = o_d_w * 1
            o_c_w = o_c_w * 1
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.05

            # no smoothness
            s_w = s_w * 0
            a_w = a_w * 0
            v_w = v_w * 0
        if level == 3:
            # learn how to reach a target position
            self.min_distance = 0.10
            d_w = d_w * 0

            # do not consider obstacle
            o_d_w = o_d_w * 5
            o_c_w = o_c_w * 1
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.15

            # no smoothness
            s_w = s_w * 0
            a_w = a_w * 0
            v_w = v_w * 0
        if level == 4:
            # learn how to reach a target position
            self.min_distance = 0.10
            d_w = d_w * 0

            # do not consider obstacle
            o_d_w = o_d_w * 10
            o_c_w = o_c_w * 1
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.30

            # no smoothness
            s_w = s_w * 1
            a_w = a_w * 1
            v_w = v_w * 1
        if level == 5:
            # learn how to reach a target position
            self.min_distance = 0.10
            d_w = d_w * 0

            # do not consider obstacle
            o_d_w = o_d_w * 0
            o_c_w = o_c_w * 1
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.30

            # no smoothness
            s_w = s_w * 5
            a_w = a_w * 5
            v_w = v_w * 5
        if level == 6:
            # learn how to reach a target position
            self.min_distance = 0.10
            d_w = d_w * 0

            # do not consider obstacle
            o_d_w = o_d_w * 0
            o_c_w = o_c_w * 1
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.30

            # no smoothness
            s_w = s_w * 10
            a_w = a_w * 10
            v_w = v_w * 10
        if level == 7:
             # learn how to reach a target position
            self.min_distance = 0.10
            d_w = d_w * 0

            # do not consider obstacle
            o_d_w = o_d_w * 0
            o_c_w = o_c_w * 1
            self.obstacle_distance_treshold = 0.5
            self.obstacle_collision_distance = 0.30

            # no smoothness
            s_w = s_w * 20
            a_w = a_w * 20
            v_w = v_w * 20

        return g_w, c_w, d_w, s_w, a_w, v_w, o_d_w, o_c_w


    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        env_state = self.env._robot_server_state_to_env_state(rs_state)
        reward = 0
        done = False
        info = {}

        level = self.get_level()
        g_w, c_w, d_w, s_w, a_w, v_w, o_d_w, o_c_w = self.get_weights(level)

        
        # Calculate distance to the target
        target_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'], rs_state['object_0_to_ref_translation_z']])
        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'], rs_state['ee_to_ref_translation_z']])
        obstacle_coord = np.array([rs_state['object_1_to_ref_translation_x'], rs_state['object_1_to_ref_translation_y'], rs_state['object_1_to_ref_translation_z']])
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        obstacle_distance = np.linalg.norm(obstacle_coord - ee_coord)
        
        joint_velocities = np.array([rs_state['base_joint_velocity'], rs_state['shoulder_joint_velocity'], rs_state['elbow_joint_velocity'], rs_state['wrist_1_joint_velocity'], rs_state['wrist_2_joint_velocity'], rs_state['wrist_3_joint_velocity']]) 

        previous_action = self.previous_action
        

        # distance weight
        x = d_w * euclidean_dist_3d
        reward += x
        self.reward_composition['distance_weight'] += x
        
        # obstacle distance 
        if obstacle_distance < self.obstacle_distance_treshold:
            x = o_d_w * (1 - obstacle_distance/self.obstacle_distance_treshold)
        else:
            x = 0
        reward += x
        self.reward_composition['obstacle_distance'] += x

        # obstacle collision
        if obstacle_distance < self.obstacle_collision_distance:
            x = o_c_w * 1
            print('obstacle distance')
        else:
            x = 0
        reward += x
        self.reward_composition['obstacle_collision']

        # smoothness 
        x = s_w * np.linalg.norm(action - previous_action)**2
        reward += x
        self.reward_composition['smoothness_weight'] += x
        # action magnitude
        x = a_w * np.linalg.norm(action)**2
        reward += x
        self.reward_composition['action_magnitude_weight'] += x
        # velocity magnitude 
        x = v_w * np.linalg.norm(joint_velocities)**2
        reward += x
        self.reward_composition['velocity_magnitude_weight'] += x


        if euclidean_dist_3d <= self.min_distance:
            # goal reached
            x = g_w * 1
            reward = x
            self.reward_composition['goal_reached_weight'] += x

            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord
            
        if rs_state['in_collision']: 
            # punishment for collision
            x = c_w * 1
            reward = x
            self.reward_composition['collision_weight'] += x

            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['target_coord'] = target_coord
        
        return reward, done, info
