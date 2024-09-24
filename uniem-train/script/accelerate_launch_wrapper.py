"""This file is to be run by the SLURM job array (defined in `train.py`)."""

import os
import subprocess
import argparse
from pathlib import Path
import random

DEFAULT_MASTER_PORT = 29500
MIN_PORT = 2000
MAX_PORT = 65000

path_to_trlx = Path(__file__).parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16")

    args, extra = parser.parse_known_args()

    slurm_job_id = int(os.getenv("SLURM_JOB_ID", random.randint(0, 100_000)))
    seed = slurm_job_id
    main_process_port = MIN_PORT + (DEFAULT_MASTER_PORT + slurm_job_id) % (MAX_PORT - MIN_PORT)

    command = [
        "accelerate", "launch",

        # arguments to accelerate.launch
        "--num_processes", str(args.num_processes),
        "--main_process_port", str(main_process_port),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), 
        "--mixed_precision", args.mixed_precision,

        # this is an executable, next lines are arguments to it
        *extra,
    ]

    print('\n> [accelerate_launch_wrapper.py] Running: ', " ".join(command), '\n')

    subprocess.run(command)