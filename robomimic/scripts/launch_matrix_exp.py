import json
import os
import shutil
import subprocess
import datetime
import time
import dateutil.tz
import argparse
from doodad.slurm.slurm_util import wrap_command_with_sbatch_matrix, SlurmConfigMatrix
import robomimic
from robomimic.config.base_config import config_factory

def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir


SINGULARITY_PRE_CMDS = [
    "export MUJOCO_GL='egl'",
    "export MKL_THREADING_LAYER=GNU",
    "export D4RL_SUPPRESS_IMPORT_ERROR='1'",
]

slurm_config_dict = {
    'partition': "russ_reserved",
    'time': "72:00:00",
    'n_gpus': 1,
    'n_cpus_per_gpu': 20,
    'mem': "62g",
    'extra_flags': "--exclude=matrix-1-[4,8,12,16],matrix-0-[24,38]",
}
slurm_config = SlurmConfigMatrix(**slurm_config_dict)

def run_on_slurm(config_path, sif_path):
    ext_cfg = json.load(open(config_path, 'r'))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    log_dir, ckpt_dir, video_dir = get_exp_dir(config)

    # Generate the command
    python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[:-1]
    command = " ".join((python_cmd, "robomimic/scripts/train.py --config ", config_path, " --log_dir ", log_dir, " --ckpt_dir ", ckpt_dir, " --video_dir ", video_dir))
    singularity_pre_cmds = " && ".join(SINGULARITY_PRE_CMDS)
    slurm_cmd = wrap_command_with_sbatch_matrix(
        f"/opt/singularity/bin/singularity exec --nv {sif_path} /bin/zsh -c \"{singularity_pre_cmds} && source ~/.zshrc && {command}\"",
        slurm_config,
        log_dir,
    )

    # Execute the command
    print(slurm_cmd)
    os.system(slurm_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Python script on Slurm with Singularity")
    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument("script_args", nargs="*", help="Arguments for the Python script")
    parser.add_argument("--sif", required=True, help="Path to the Singularity .sif file")
    
    args = parser.parse_args()

    run_on_slurm(args.script, args.script_args, args.sif, args.conda_env)
