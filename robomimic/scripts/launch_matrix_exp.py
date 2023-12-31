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


slurm_additional_parameters = {
    "partition": "deepaklong",
    "time": "00:03:00",
    "gpus": 1,
    "cpus_per_gpu": 16,
    "mem": "100g",
    #"exclude": "matrix-1-[4,8,10,12,16],matrix-0-[24,38]",
    "nodelist": "grogu-1-3"
}


class WrappedCallable(submitit.helpers.Checkpointable):
    def __init__(self, output_dir, sif_path, python_path, file_path, config_path):
        self.output_dir = output_dir
        self.sif_path = sif_path
        self.python_path = python_path
        self.file_path = file_path
        self.config_path = config_path
        self.p = None

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
            self.output_dir, self.sif_path, self.python_path, self.file_path, self.config_path
        )
        ckpt_dir = os.path.join(self.output_dir, "models")
        checkpoint_path = os.path.join(ckpt_dir, "model_latest.pth")
        print("RESUBMITTING")
        return submitit.helpers.DelayedSubmission(wrapped_callable, checkpoint_path)


def run_on_slurm(config_path, sif_path):
    ext_cfg = json.load(open(config_path, "r"))
    config = config_factory(ext_cfg["algo_name"])
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base algo config
    with config.values_unlocked():
        config.update(ext_cfg)
    output_dir = get_exp_dir(config)

    # Generate the command
    python_cmd = subprocess.check_output("which python", shell=True).decode("utf-8")[
        :-1
    ]
    executor = submitit.AutoExecutor(
        folder=output_dir + "/%j",
    )
    executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)
    # absolute path to robomimic/scripts/train.py
    file_path = os.path.join(robomimic.__path__[0], "scripts/train.py")
    wrapped_callable = WrappedCallable(
        output_dir, sif_path, python_cmd, file_path, config_path
    )
    job = executor.submit(wrapped_callable, None)


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
