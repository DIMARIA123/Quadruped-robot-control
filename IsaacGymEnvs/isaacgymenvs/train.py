from datetime import datetime

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import isaacgym # must import isaacgym before pytorch
import isaacgymenvs
import gym
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.rlgames_utils import (
    RLGPUEnv,
    RLGPUAlgoObserver,
    MultiObserver,
)

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder, a2c_continuous
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import models

@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def initialize_and_run_rl(cfg: DictConfig):

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"train_{time_str}"

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    observers = [RLGPUAlgoObserver()]

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)

        runner.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        
        model_builder.register_model('continuous_a2c',lambda network, **kwargs: models.ModelA2CContinuous(network))
        model_builder.register_model('continuous_a2c_logstd',lambda network, **kwargs: models.ModelA2CContinuousLogStd(network))
        model_builder.register_model('soft_actor_critic',lambda network, **kwargs: models.ModelSACContinuous(network))
        model_builder.register_model('central_value',lambda network, **kwargs: models.ModelCentralValue(network))
        
        model_builder.register_network('actor_critic', lambda **kwargs: network_builder.A2CBuilder())
        model_builder.register_network('resnet_actor_critic',lambda **kwargs: network_builder.A2CResnetBuilder())
        model_builder.register_network('rnd_curiosity', lambda **kwargs: network_builder.RNDCuriosityBuilder())
        model_builder.register_network('soft_actor_critic', lambda **kwargs: network_builder.SACBuilder())
        return runner

    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    initialize_and_run_rl()
