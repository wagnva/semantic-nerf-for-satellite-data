#!/bin/bash
path=$1
session=$2
env_n=$3

# checking if enough free gpus are available (based on passed ids)
shift 3
if [ "$#" -lt 3 ]; then
    echo "Three free GPUs are required for this script. You can adapt the tmux.sh script to your setup"
    exit 1
fi
gpu1=$1
gpu2=$2
gpu3=$3

# start tmux session
tmux new-session -d -s $session
tmux setw -g mouse on

# train eval
tmux rename-window -t $session:0 'train'
tmux send-keys -t $session:0 "conda activate ${env_n}; python -m eval.eval_nerf ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu1} --device_req_free=False --split=train" C-m
# test eval
tmux new-window -t $session:1 -n 'test'
tmux send-keys -t $session:1 "conda activate ${env_n}; python -m eval.eval_nerf ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu2} --device_req_free=False --split=test" C-m
# pointclouds
tmux new-window -t $session:2 -n 'pcs'
tmux send-keys -t $session:2 "conda activate ${env_n}; python -m eval.extract_pointcloud ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu3} --device_req_free=False --split=train --max_items=3" C-m

# start with first window selected
tmux select-window -t $session:0