import gymnasium as gym
from grpc import RpcError
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError

class ExceptionHandling(gym.Wrapper):

    def step(self, action):
        try:
            observation, reward, done, truncated, info = self.env.step(action)
            return observation, reward, done, truncated, info
        except (RpcError, InvalidStateError, RobotServerError) as e:
            print('Error occurred while calling the step function. Restarting Robot server ...')
            self.env.restart_sim()
            return self.env.observation_space.sample(), 0, True, False, {"Exception":True, "ExceptionType": e}

    def reset(self, **kwargs):
        for i in range(5):
            try:
                return self.env.reset(**kwargs)
            except (RpcError, InvalidStateError, RobotServerError):
                print('Error occurred while calling the reset function. Restarting Robot server ...')
                self.env.restart_sim()
        raise Exception("Failed 5 tentatives to reset environment.")
