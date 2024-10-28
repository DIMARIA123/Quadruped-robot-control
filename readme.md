
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

Currently, the environment is set up using IsaacGymEnv, and training is conducted with the built-in PPO algorithm from the rl-game library, which supports the asymmetric actor-critic variant. The training can be run with the following command:
   ```bash
   cd IsaacGymEnvs/isaacgymenvs
   python train.py
   ```
IsaacGymEnv uses the Hydra library for parameter configuration, and to maintain consistency, this project also utilizes Hydra for parameter management. The top-level configuration file is located at 
   IsaacGymEnvs/isaacgymenvs/cfg/config.yaml. 
Configuration files for the environment and the reinforcement learning algorithm can be found at:
   IsaacGymEnvs/isaacgymenvs/cfg/task/Go2.yaml
   IsaacGymEnvs/isaacgymenvs/cfg/train/Go2PPO.yaml 
Each configuration file includes detailed comments; for more information, please refer to the respective documentation.

---

## Further Documentation

For information on the RL-Game library and the IsaacGym library, please refer to the ref folder, or visit the official GitHub repository. For a more detailed explanation of the code, please refer to the project report.

