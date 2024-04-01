#!/bin/bash

export PATH="/netscratch/georgiou/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)" && conda activate ./temp-spsc
#pip install -r spkanon_eval/requirements.txt

"$@"