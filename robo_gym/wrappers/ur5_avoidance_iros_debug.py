import gym 

class UR5AvoidanceIrosDebugWrapper(gym.Wrapper):
     def _get_desired_joint_positions(self) -> np.array:
        
        return self.fixed_joint_positions

    def reset(self, fixed_joint_positions, **kwargs):
        self.fixed_joint_positions = copy.deepcopy(fixed_joint_positions)
        return self.env.reset(**kwargs)


