## Setup

environment: Ubuntu 20.04.6 LT Cuda12.1
Yaml file is the ‘environment.yaml‘ in curent content

### perpare legged_gym，isaacgym

```sh
conda activate ME5418-group23
cd rsl_rl-1.0.2 && pip install -e .
cd .. 
cd isaacgym/python && pip install -e .
cd .. 
cd ..
cd legged_gym && pip install -e .
cd ..
```

## demo running

```sh
conda activate legged_robot_parkour
cd legged_gym/legged_gym/tests

python test_env.py --num_envs=1
```

If there has error:
```sh
AttributeError: module 'numpy' has no attribute 'float'.
```

solution: search all 'np.float', change it to 'float'

