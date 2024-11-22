import numpy as np
import os
import torch

from isaacgym import gymtorch, gymapi, terrain_utils

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask

class Go2(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
    
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
 
        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"] # Whether Allow Knee Contact with the Ground
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt
        
        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) 
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dofs)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands[..., 1]
        self.commands_x = self.commands[..., 0]
        self.commands_yaw = self.commands[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.cfg["env"]["defaultJointAngles"][name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    # Three required functions (see in RL-game rep doc) 
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        actions_scaled = actions * 0.25
        self.action_torques = 50.*(actions_scaled + self.default_dof_pos - self.dof_pos) - 2.*self.dof_vel
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.action_torques))

    def post_physics_step(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.progress_buf += 1

        # prepare quantities
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.compute_observations()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]


    # functions called in the create_sim function
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/go2/urdf/go2.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        go2_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        body_names = self.gym.get_asset_rigid_body_names(go2_asset)
        self.dof_names = self.gym.get_asset_dof_names(go2_asset)
        dof_props = self.gym.get_asset_dof_properties(go2_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(go2_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if "foot" in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)


        for i in range(self.num_dofs):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
        

        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(len(dof_props)):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            self.gym.set_asset_rigid_shape_properties(go2_asset, rigid_shape_props_asset)
            actor_handle = self.gym.create_actor(env_handle, go2_asset, start_pose, "go2", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)



        penalize_contacts_on = ["thigh", "calf"]
        penalized_contact_names = []
        for name in penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        termination_contact_names = []
        for name in ["base"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # functions called in the post_physics_step function
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.progress_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        base_z = self.root_states[:, 2]
        z_threshold_buff = base_z < .1
        self.reset_buf |= z_threshold_buff

    def compute_reward(self):

        self.rew_buf[:] = 0.
        reward_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * -0.01 * self.dt
        reward_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -0.01 * self.dt
        reward_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -0.1 * self.dt

        reward_torques = torch.sum(torch.square(self.torques), dim=1) * -0.02 * self.dt
        reward_dof_vel = torch.sum(torch.square(self.dof_vel), dim=1) * -0.02 * self.dt
        reward_dof_acc = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1) * -0.02 * self.dt
        reward_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * -0.01 * self.dt
        reward_collision = torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) * -1. * self.dt

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward_tracking_lin_vel = torch.exp(-lin_vel_error/0.25) * 2.0 * self.dt

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        reward_tracking_ang_vel = torch.exp(-ang_vel_error/0.25) * 0.5 * self.dt

        reward_stumble = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1) * 0. * self.dt
        
        reward_stand_still = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1) * -0.1 * self.dt



        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt

        reward_feet_air_time = torch.sum(self.feet_air_time, dim = 1) * self.dt

        self.rew_buf = reward_lin_vel_z + reward_ang_vel_xy + reward_orientation + reward_torques + reward_dof_vel\
                        + reward_dof_acc + reward_action_rate + reward_collision + reward_tracking_lin_vel + reward_tracking_ang_vel +\
                        reward_stumble + reward_stand_still + reward_feet_air_time

        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)


    def compute_observations(self):
        
        # Automatic updates due to pointer mapping mechanism
        base_quat = self.root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10]) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13]) * self.ang_vel_scale
        projected_gravity = quat_rotate(base_quat, self.gravity_vec)
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale
        commands_scaled = self.commands*torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], requires_grad=False, device=self.commands.device)

        self.obs_buf = torch.cat((  base_lin_vel,
                                    base_ang_vel,
                                    projected_gravity,
                                    commands_scaled,
                                    dof_pos_scaled,
                                    self.dof_vel * 0.05,
                                    self.actions,

                                    ),dim=-1)

    # call in init and post_physics_step
    def reset_idx(self, env_ids):

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dofs), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


class Cus_Terrain:

    def __init__(self, cfg) -> None:

        self.horizontal_scale = cfg['horizontal_scale']  # [m]
        self.vertical_scale = cfg['vertical_scale']  # [m]
        self.env_length = cfg['env_length']
        self.env_width = cfg['env_width'] # The length and width of the sub-terrain(m).
        self.proportions = cfg['proportions'] # The proportion of different types of terrain.
        self.num_rows = cfg['num_rows'] # number of terrain rows (levels)
        self.num_cols = cfg['num_cols'] # number of terrain cols (types)
        self.num_sub_terrains = self.num_rows * self.num_cols # The total number of sub-terrain.
        self.env_origins = np.zeros((self.num_rows, self.num_cols, 3)) # The origin coordinates of each sub-terrain.
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale) 
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale) # The length and width of the sub-terrain(pixel).
        self.border_size = cfg['nborder_size'] # Boundary plane width(m).
        self.border = int(self.border_size/self.horizontal_scale) # Boundary plane width(pixel).
        self.tot_cols = int(self.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.num_rows * self.length_per_env_pixels) + 2 * self.border # The total length and width of the terrain.
        self.slope_treshold = cfg['slope_treshold']
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16) # heightfield
        self._make_map()
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.horizontal_scale,
                                                                                            self.vertical_scale,
                                                                                            self.slope_treshold)
    def _make_map(self):
        for j in range(self.num_cols):
            for i in range(self.num_rows):
                difficulty = i / self.num_rows
                choice = j / self.num_cols + 0.001

                terrain = self._make_terrain(choice, difficulty)
                self._add_terrain_to_map(terrain, i, j)

    def _make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        return terrain    
    
    def _add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]