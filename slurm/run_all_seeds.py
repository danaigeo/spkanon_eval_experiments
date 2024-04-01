"""
Run the given config file for all seeds within the given range.
"""


import os
import sys
from argparse import ArgumentParser
import yaml
from time import time


CONFIGS_DIR = "/netscratch/georgiou/spkanon/logs/configs"
SCRIPT = "/netscratch/georgiou/spkanon/slurm/run-knnvc.sh"


def main(args):
    """
    1. Create cfg for each seed and dump them in a tmp folder.
    2. Run the jobs in a tmux session, each job in a separate window.
    """
    timestamp = int(time())
    # create the tmp folder to store the config files
    cfg_dir = os.path.join(CONFIGS_DIR, f"configs_{timestamp}")
    os.mkdir(cfg_dir)
    # load the config file
    config = yaml.full_load(open(args.config))
    # create the tmux session
    session_name = f"seeds{args.start}-{args.end}_{timestamp}"
    cmd = f"tmux new-session -d -s {session_name}"
    os.system(cmd)
    # create the config files and run the jobs, each in a separate window
    for i in range(args.start, args.end + 1):
        config["seed"] = i*100
        dump_file = os.path.join(cfg_dir, f"config_seed{i}.yaml")
        yaml.dump(config, open(dump_file, "w"), sort_keys=False)
        #cmd = f"bash {SCRIPT} {args.config} {dump_file}"
        cmd = f"tmux new-window -t {session_name}: -n seed{i} bash {SCRIPT } {args.config} {dump_file}"
        os.system(cmd)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--start", type=int, help="Start seed index", default=0)
    parser.add_argument("--end", type=int, help="End seed index", default=5)
    parser.add_argument(
        "--partition", type=str, help="Cluster partition", default="RTXA6000"
    )
    parser.add_argument("--n_nodes", type=int, help="Number of nodes", default=1)
    parser.add_argument("--n_devices", type=int, help="Number of devices", default=1)
    args = parser.parse_args()
    main(args)