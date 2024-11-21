
# Quadruped-robot-control
ME5418 Project: Motion Control for Quadruped Robot Based on Proprioception

## Environment Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/DIMARIA123/Quadruped-robot-control.git
   ```

2. **Create Conda Environment**  
   ```bash
   conda env create -f environment.yml
   conda activate ME5418_group23
   ```
 
3. **Install Isaacgym library**  
   ```bash
   cd isaacgym/python
   pip install -e .
   ```
   
4. **Install Isaacgymenvs library**  
   ```bash
   cd IsaacGymEnvs
   pip install -e .
   ```
   
---

## Training

The current environment is set up using IsaacGymEnv, and training is conducted with the A2C algorithm, which supports the continuous action space variant of A2C. The specific configuration is as follows:
* Reinforcement Learning Algorithm: a2c_continuous
* Model: continuous_a2c_logstd
* Network Architecture: actor_critic

The training can be run with the following command:
   ```bash
   cd IsaacGymEnvs/isaacgymenvs
   python custrain.py
   ```
IsaacGymEnv uses the Hydra library for parameter configuration, and to maintain consistency, this project also utilizes Hydra for parameter management. The top-level configuration file is located at 
* IsaacGymEnvs/isaacgymenvs/cfg/cusconfig.yaml. 

Configuration files for the environment and the reinforcement learning algorithm can be found at:
* IsaacGymEnvs/isaacgymenvs/cfg/task/Go2.yaml
* IsaacGymEnvs/isaacgymenvs/cfg/train/Go2PPO.yaml 

If you want to modify configuration parameters, please edit the corresponding configuration file directly. Alternatively, you can override parameters by adding them as arguments when running python train.py --parameter_name value (although I personally do not recommend this approach).

---

## Further Documentation

For information on the RL-Game library and the IsaacGym library, please refer to the ref folder, or visit the official GitHub repository. For a more detailed explanation of the code, please refer to the project report.

