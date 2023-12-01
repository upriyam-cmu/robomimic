import os
import subprocess
import datetime
import dateutil.tz
import argparse
from doodad.slurm.slurm_util import wrap_command_with_sbatch_matrix, SlurmConfigMatrix


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
    # Create log directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime(f"%Y_%m_%d_%H_%M_%S_{now.microsecond}")
    cwd = os.getcwd()
    log_dir = os.path.join(cwd, f"logs/{timestamp}")
    os.makedirs(log_dir)

    # Generate the command
    python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[:-1]
    command = " ".join((python_cmd, "--config ", config_path))
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
