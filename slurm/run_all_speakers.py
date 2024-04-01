"""
Run the given config file for all target speakers within the given range.
"""


import os
import sys
from argparse import ArgumentParser
import yaml
from time import time


CONFIGS_DIR = "/netscratch/georgiou/spkanon/logs"
SCRIPT = "/netscratch/georgiou/spkanon/slurm/run-stargan.sh"


def main(args):
    """
    1. Create cfg for each target speaker and dump them in a tmp folder.
    2. Run the jobs in a tmux session, each job in a separate window.
    3.
    """
    timestamp = int(time())
    # create the tmp folder to store the config files
    cfg_dir = os.path.join(CONFIGS_DIR, f"configs_{timestamp}")
    os.mkdir(cfg_dir)
    # load the config file
    config = yaml.full_load(open(args.config))
    # create the tmux session
    session_name = f"spk{args.start}-{args.end}_{timestamp}"
    cmd = f"tmux new-session -t {session_name} -d"
    os.system(cmd)
    # dump the arguments into a string
    run_args = f"{args.partition} {args.n_nodes} {args.n_devices}"
    # create the config files and run the jobs, each in a separate window
    for i in range(args.start, args.end + 1):
        config["data"]["config"]["max_speakers"] = i
        dump_file = os.path.join(cfg_dir, f"config_spk{i}.yaml")
        yaml.dump(config, open(dump_file, "w"))
        window = f"spk{i}"
        os.system(f"tmux new-window -t {session_name} -n {window}")
        run_cmd = f"\"bash {SCRIPT} {run_args} {dump_file}\" Enter"
        os.system(f"tmux send-keys -t {session_name}:{window} {run_cmd}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to config file"
    )
    parser.add_argument(
        "--start", type=int, help="Start speaker index", default=0, nargs="?"
    )
    parser.add_argument(
        "--end", type=int, help="End speaker index", default=2, nargs="?"
    )
    parser.add_argument(
        "--partition", type=str, help="Cluster partition", default="RTXA6000", nargs="?"
    )
    parser.add_argument(
        "--n_nodes", type=int, help="Number of nodes", default=1, nargs="?"
    )
    parser.add_argument(
        "--n_devices", type=int, help="Number of devices", default=1, nargs="?"
    )
    args = parser.parse_args()
    main(args)  