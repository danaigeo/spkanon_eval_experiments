#!/bin/bash

PARTITION=RTXA6000 \

#CONFIG=/netscratch/georgiou/spkanon/trainers/knnvc.yaml

#Check if argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 CONFIG_FILE_PATH DUMP_FILE_PATH"
    exit 1
fi

CONFIG=$1
DUMP_FILE=$2


srun \
  --job-name=$(basename "$CONFIG" .yaml) \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
  --container-mounts=/ds:/ds,/netscratch/$USER:/netscratch/$USER \
  --container-workdir=/netscratch/$USER/spkanon \
  --mem=100G \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=10 \
  --gpus-per-task=1 \
  --partition=$PARTITION \
  --kill-on-bad-exit \
  /netscratch/georgiou/slurm_commands/spkanon/activate_venv.sh python spkanon_eval/run.py --config "$DUMP_FILE"