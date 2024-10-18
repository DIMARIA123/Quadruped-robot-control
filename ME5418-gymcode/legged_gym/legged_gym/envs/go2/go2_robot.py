from .go2_config import Go2RoughCfg
from  legged_gym.envs.base.legged_robot import LeggedRobot

class Go2Robot(LeggedRobot):
    cfg: Go2RoughCfg
