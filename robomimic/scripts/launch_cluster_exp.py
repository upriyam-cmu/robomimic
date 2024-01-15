import json
import os
import subprocess
import datetime
import time
import argparse
import robomimic
from robomimic.config.base_config import config_factory
import submitit


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
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y%m%d%H%M%S")

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name, time_str)
    return base_output_dir


# slurm_additional_parameters = {
#     "partition": "all",
#     "time": "00:07:00",
#     "gpus": 1,
#     "cpus_per_gpu": 16,
#     "mem": "100g",
#     # "exclude": "grogu-1-14, grogu-1-19, grogu-0-24, grogu-1-[9,24,29], grogu-3-[1,3,5,9,11,25,27], grogu-3-[15,17,19,21,23], grogu-3-29", # 5000/6000
#     "exclude": "grogu-3-[1,3,5,9,11,25,27], grogu-3-[15,17,19,21,23], grogu-3-29", # 5000/6000 + 2080, 3080
# }

slurm_additional_parameters = {
    "partition": "russ_reserved",
    "time": "3-00:00:00",
    "gpus": 1,
    "cpus_per_gpu": 20,
    "mem": "62g",
    "exclude": "matrix-1-[4,6,8,10,12,16,18,20],matrix-0-[24,34,38]", # Throw out non-rtx
}


class WrappedCallable(submitit.helpers.Checkpointable):
    def __init__(self, output_dir, sif_path, python_path, file_path, config_path, start_from_checkpoint):
        self.output_dir = output_dir
        self.sif_path = sif_path
        self.python_path = python_path
        self.file_path = file_path
        self.config_path = config_path
        self.p = None
        self.start_from_checkpoint = start_from_checkpoint

    def __call__(self, checkpoint_path=None):
        """
        """
        # launch function in a singularity container:
        singularity_path = "singularity"
        output_dir = self.output_dir
        if checkpoint_path is not None:
            output_dir = None # get output_dir from ckpt
        cmd = f"{singularity_path} exec --nv {self.sif_path} {self.python_path} {self.file_path} --config {self.config_path} \
            --output_dir {output_dir} --agent {checkpoint_path}"
        if self.start_from_checkpoint:
            cmd += " --start_from_checkpoint"
        print(cmd)
        self.p = subprocess.Popen(cmd, shell=True)
        while True:
            pass

    def checkpoint(self, checkpointpath: str) -> submitit.helpers.DelayedSubmission:
        print("sending checkpoint signal")
        import signal

        os.kill(self.p.pid, signal.SIGUSR1)
        print("wait for 30s")
        time.sleep(30)
        print("setup new callable")
        wrapped_callable = WrappedCallable(
            self.output_dir, self.sif_path, self.python_path, self.file_path, self.config_path, start_from_checkpoint=False
        )
        ckpt_dir = os.path.join(self.output_dir, "models")
        checkpoint_path = os.path.join(ckpt_dir, "model_latest.pth")
        print("RESUBMITTING")
        return submitit.helpers.DelayedSubmission(wrapped_callable, checkpoint_path)


def run_on_slurm(config_path, sif_path, checkpoint_path=None):
    ext_cfg = json.load(open(config_path, "r"))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    output_dir = get_exp_dir(config)

    # Generate the command
    python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[:-1]
    executor = submitit.AutoExecutor(
        folder=os.path.join(output_dir, "%j"),
        slurm_max_num_timeout=1000,
    )
    job_name = config.experiment.name
    slurm_additional_parameters["job_name"] = job_name
    executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)
    # absolute path to robomimic/scripts/train.py
    file_path = os.path.join(robomimic.__path__[0], "scripts/train.py")
    start_from_checkpoint = True if checkpoint_path is not None else False
    wrapped_callable = WrappedCallable(
        output_dir, sif_path, python_cmd, file_path, config_path, start_from_checkpoint
    )
    # basically if we take in a checkpoint path, we want to start from that checkpoint
    job = executor.submit(wrapped_callable, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Python script on Slurm with Singularity"
    )
    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument(
        "script_args", nargs="*", help="Arguments for the Python script"
    )
    parser.add_argument(
        "--sif", required=True, help="Path to the Singularity .sif file"
    )

    args = parser.parse_args()

    run_on_slurm(args.script, args.script_args, args.sif, args.conda_env)
