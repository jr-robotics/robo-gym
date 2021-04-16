import gym
import robo_gym
import pytest



envs = [
    ('EmptyEnvironmentURSim-v0', 'ur5'),
    ('EndEffectorPositioningURSim-v0', 'ur5'),
    ('MovingBoxTargetURSim-v0', 'ur5'),
    ('IrosEnv03URTrainingSim-v0', 'ur5'),
    ('IrosEnv03URTestFixedSplinesSim-v0', 'ur5')
]



@pytest.mark.parametrize('env_name, ur_model', envs)
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
