import os

import gymnasium as gym
import robo_gym
import pytest

@pytest.fixture(scope='module')
def env(request):
    ip = os.environ.get("ROBOGYM_SERVERS_HOST", 'robot-servers')
    env = gym.make('NoObstacleNavigationMir100Sim-v0', ip=ip)
    yield env
    env.kill_sim()

@pytest.mark.commit 
def test_initialization(env):
    env.reset()
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _, _ = env.step(action)

    assert env.observation_space.contains(observation)