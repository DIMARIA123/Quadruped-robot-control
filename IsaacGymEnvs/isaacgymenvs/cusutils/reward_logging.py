import pandas as pd

reward_log = pd.DataFrame(columns=[
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_torques",
    "reward_dof_vel",
    "reward_dof_acc",
    "reward_action_rate",
    "reward_collision",
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_stand_still",
    "reward_feet_air_time",
    "total_reward"
])

def log_rewards(
    reward_lin_vel_z, reward_ang_vel_xy, reward_orientation, reward_torques,
    reward_dof_vel, reward_dof_acc, reward_action_rate, reward_collision,
    reward_tracking_lin_vel, reward_tracking_ang_vel, reward_stand_still,
    reward_feet_air_time, total_reward
):
    global reward_log

    reward_values = {
        "reward_lin_vel_z": reward_lin_vel_z.mean().item(),
        "reward_ang_vel_xy": reward_ang_vel_xy.mean().item(),
        "reward_orientation": reward_orientation.mean().item(),
        "reward_torques": reward_torques.mean().item(),
        "reward_dof_vel": reward_dof_vel.mean().item(),
        "reward_dof_acc": reward_dof_acc.mean().item(),
        "reward_action_rate": reward_action_rate.mean().item(),
        "reward_collision": reward_collision.mean().item(),
        "reward_tracking_lin_vel": reward_tracking_lin_vel.mean().item(),
        "reward_tracking_ang_vel": reward_tracking_ang_vel.mean().item(),
        "reward_stand_still": reward_stand_still.mean().item(),
        "reward_feet_air_time": reward_feet_air_time.mean().item(),
        "total_reward": total_reward.mean().item(),
    }
    reward_log = reward_log.append(reward_values, ignore_index=True)