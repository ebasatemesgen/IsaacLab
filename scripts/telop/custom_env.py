import math
import torch
from dataclasses import MISSING
from typing import Literal
import argparse  

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG 
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# from omniverse_sim import args_cli
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import MassPropertiesCfg, CollisionPropertiesCfg


base_command = {}


def constant_commands(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    global base_command
    """The generated command from the command generator."""
    tensor_lst = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    for i in range(env.num_envs):
        tensor_lst[i] = torch.tensor(base_command[str(i)], dtype=torch.float32, device=env.device)
    return tensor_lst


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        debug_vis=False,
    )
    
    # if args_cli.terrain == "flat":
    
    #     # flat terrain
    #     terrain = TerrainImporterCfg(
    #         prim_path="/World/ground",
    #         terrain_type="plane",
    #         debug_vis=False,
    #     )
    # elif args_cli.terrain == "rough":
    #     terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #     friction_combine_mode="multiply",
    #     restitution_combine_mode="multiply",
    #     static_friction=1.0,
    #     dynamic_friction=1.0,
    #     ),
    #     debug_vis=False,
    #     )

    # robots
    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


    # Add the missing contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


    # Add a pushable box using a built-in cube asset from Isaac Sim
    pushable_box: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/PushableBox",  # where the box will be spawned
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/basic_block.usd",  # using a built-in block asset
            scale=(10, 10, 10),  # increase scale to make it bigger
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False
            ),
            mass_props=MassPropertiesCfg(mass=5),  # set mass to 0.5 kg (half a kilo)
            collision_props=CollisionPropertiesCfg(),  # use default collision properties (optional)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.5)  # initial position (x, y, z)
        )
    )
    # Add a permanent goal marker (a flat cylinder)
    goal_circle: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GoalCircle",  # the prim path where the goal marker is spawned
        spawn=sim_utils.CylinderCfg(
            radius=0.5,          # adjust to match the desired goal size (here, a diameter of 3)
            height=0.1,          # a low height (e.g. 2 units tall)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # yellow color for visibility
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(5.0, 5.0, 0.0)  # center the goal within a 10×10 region (z is half the height so it sits on the ground)
        )
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (1920, 1080)

    origin_type: Literal["world", "env", "asset_root"] = "world"

    env_index: int = 0

    asset_name: str | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func= mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg()
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        # self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class UnitreeGo2CustomEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

