import gym
import robo_gym
import itertools
import pytest


envs = [
    # 'EmptyEnvironmentURSim-v0',
    # 'EndEffectorPositioningURSim-v0',
    # 'MovingBoxTargetURSim-v0',
    'IrosEnv03URTrainingSim-v0', 
    'IrosEnv03URTestFixedSplinesSim-v0'
]

ur_models = ["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"]

envs_ur_model_combinations = list(itertools.product(envs, ur_models))

@pytest.mark.parametrize('env_name, ur_model', envs_ur_model_combinations)
@pytest.mark.filterwarnings('ignore:UserWarning')
def test_env_initialization(env_name, ur_model):
    env = gym.make(env_name, ip='robot-servers', ur_model=ur_model)

    env.reset()
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

    assert env.observation_space.contains(observation)

    env.kill_sim()
    env.close()
