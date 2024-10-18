from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry

import torch


def test_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    while True:
        action = dummy_policy()
        _, _, _, _, _ = env.step(action)

def dummy_policy():
    return torch.empty((1,12)).uniform_(-1.0, 1.0)

if __name__ == '__main__':
    args = get_args()
    test_env(args)
