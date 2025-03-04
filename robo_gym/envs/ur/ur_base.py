from envs.manipulator.manipulator_base import ManipulatorBaseEnv
from envs.robogym_env import RoboGymEnv
from utils.ur_utils import UR


class URBaseEnv2(ManipulatorBaseEnv):

    KW_UR_MODEL_KEY = "ur_model"

    def __init__(self, **kwargs):

        # prepare UR model depending on ur_model and set it in the kwargs
        robot_model = UR(model_key=self.ur_model_key)
        kwargs[RoboGymEnv.KW_ROBOT_MODEL_OBJECT] = robot_model

        # super initializer will also set robot_model as self._robot_model
        super().__init__(**kwargs)

    @property
    def ur_model_key(self) -> str:
        return self._config.get(URBaseEnv2.KW_UR_MODEL_KEY)
