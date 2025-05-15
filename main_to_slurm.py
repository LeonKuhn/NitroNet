# Copyright (c) 2025 Leon Kuhn
#
# This file is part of a software project licensed under a Custom Non-Commercial License.
#
# You may use, modify, and distribute this code for non-commercial purposes only.
# Use in academic or scientific research requires prior written permission.
#
# For full license details, see the LICENSE file or contact l.kuhn@mpic.de


import uuid
import tomli
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("conda_env", type=str, help="conda environment to use")
parser.add_argument("config_file", type=str, help="config_file to use")
args = parser.parse_args()

nicename = args.config_file.split("/")[-1].split(".")[0]

with open(args.config_file, mode="rb") as fp:
    c = tomli.load(fp)  # Load config file
    conf = {**c["main"]["slurm"], **c}
    del c

os.makedirs(f"{conf['general']['log_dir']}", exist_ok=True)

submit_script_preprocessing = f"""#!/bin/bash -l
#SBATCH -o {conf["general"]["log_dir"]}main.%A_%a.out
#SBATCH -e {conf["general"]["log_dir"]}main.%A_%a.err
#SBATCH -D ./
#SBATCH -J nitronet_main_{nicename}
#SBATCH --array=1-{conf["--array"]}
#SBATCH --cpus-per-task={conf["--cpus-per-task"]}
#SBATCH --mem={conf["--mem"]}
#SBATCH --time={conf["--time"]}
#SBATCH --gres=gpu:{conf["--gres=gpu"]}
conda activate {args.conda_env}
# module load rocm/6.3
python main.py {args.config_file} $SLURM_ARRAY_TASK_ID
"""

id = uuid.uuid4().hex[:6]
submission_script_name = f'main_{id}.sh'
with open(submission_script_name, 'w') as f:
    f.write(submit_script_preprocessing)

command = f"sbatch {submission_script_name}"
print("Submitting:\n", command)
os.system(command)

os.system(f"mv {submission_script_name} {conf['general']['log_dir']}.")
