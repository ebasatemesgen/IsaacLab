import argparse

from isaaclab.app import AppLauncher

# import cli_args  
import time
import os
import threading
# argparse arguments
parser = argparse.ArgumentParser(description="Policy inference on Go2 robot in a flat environment.")
# parser.add_argument("--checkpoint", type=str, help="Path to Go2 model checkpoint exported as jit.", required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse arguments

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--custom_env", type=str, default="office", help="Setup the environment")
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument("--terrain", type=str, default="rough", help="Setup the robot")
parser.add_argument("--robot_amount", type=int, default=1, help="Setup the robot amount")

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import omni


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)


"""Rest everything follows."""
import gymnasium as gym
import torch
import carb


from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab.sim as sim_utils
import omni.appwindow
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


from custom_env import UnitreeGo2CustomEnvCfg
import custom_env



def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_env.base_command["0"] = [0, 0, -1]

            if len(custom_env.base_command) > 1:
                if event.input.name == 'I':
                    custom_env.base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_env.base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_env.base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_env.base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_env.base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_env.base_command)):
                custom_env.base_command[str(i)] = [0, 0, 0]
    return True


def setup_custom_env():
    try:
        if (args_cli.custom_env == "warehouse" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if (args_cli.custom_env == "office" and args_cli.terrain == 'flat'):
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office.usd")
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))
    except:
        print("Error loading custom environment. You should download custom envs folder from: https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing")


def cmd_vel_cb(msg, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_env.base_command[str(num_robot)] = [x, y, z]




def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_env.base_command[str(i)] = [0, 0, 0]
def run_sim():
    
    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Play with RSL-RL agent."""
    # parse configuration
    print("Play with RSL-RL agent")
    env_cfg = UnitreeGo2CustomEnvCfg()
    

    env_cfg.scene.num_envs = 1

        
    specify_cmd_for_robots(env_cfg.scene.num_envs)


    unitree_go2_agent_cfg = {
            'seed': 42, 
            'device': 'cuda', 
            'num_steps_per_env': 24, 
            'max_iterations': 15000, 
            'empirical_normalization': False, 
            'policy': {
                'class_name': 'ActorCritic', 
                'init_noise_std': 1.0, 
                'actor_hidden_dims': [512, 256, 128], 
                'critic_hidden_dims': [512, 256, 128], 
                'activation': 'elu'
                }, 
            'algorithm': {
                'class_name': 'PPO', 
                'value_loss_coef': 1.0, 
                'use_clipped_value_loss': True, 
                'clip_param': 0.2, 
                'entropy_coef': 0.01, 
                'num_learning_epochs': 5, 
                'num_mini_batches': 4, 
                'learning_rate': 0.001, 
                'schedule': 'adaptive', 
                'gamma': 0.99, 
                'lam': 0.95, 
                'desired_kl': 0.01, 
                'max_grad_norm': 1.0
            }, 
            'save_interval': 50, 
            'experiment_name': 'unitree_go2_rough', 
            'run_name': '', 
            'logger': 'tensorboard', 
            'neptune_project': 'orbit', 
            'wandb_project': 'orbit', 
            'resume': False, 
            'load_run': '.*', 
            'load_checkpoint': 'model_.*.pt'
            }



    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()

    setup_custom_env()
    
    start_time = time.time()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
 
    env.close()

if __name__ == "__main__":
    run_sim()