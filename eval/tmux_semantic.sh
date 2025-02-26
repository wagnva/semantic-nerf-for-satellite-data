#!/bin/bash
path=$1
session=$2
env_n=$3

# checking if enough free gpus are available (based on passed ids)
shift 3
if [ "$#" -lt 5 ]; then
    echo "Five free GPUs are required for this script. You can adapt the tmux.sh script to your setup"
    exit 1
fi
gpu1=$1
gpu2=$2
gpu3=$3
gpu4=$4
gpu5=$5

exit 1

tmux new-session -d -s $session
tmux rename-window -t $session:0 'e_train'
tmux setw -g mouse on

tmux send-keys -t $session:0 "conda activate ${env_n}; python -m eval.eval_nerf ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu1} --device_req_free=False --split=train" C-m
tmux new-window -t $session:1 -n 'test'
tmux send-keys -t $session:1 "conda activate ${env_n}; python -m eval.eval_nerf ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu2} --device_req_free=False --split=test; python -m semantic.study.eval ${path} --device=${gpu2} --device_req_free=False --split=test; python -m semantic.run_visualizer ${path} --device=${gpu2} --device_req_free=False --split=test --save_png=True" C-m

tmux new-window -t $session:2 -n 's_train'
tmux send-keys -t $session:2 "conda activate ${env_n}; python -m semantic.study.eval ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu3} --device_req_free=False --split=train" C-m

# pointclouds
tmux new-window -t $session:3 -n 'pcs'
tmux send-keys -t $session:3 "conda activate ${env_n}; python -m eval.extract_pointcloud ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu4} --device_req_free=False --split=train --max_items=3" C-m
# visualizer
tmux new-window -t $session:4 -n 'viz'
tmux send-keys -t $session:4 "conda activate ${env_n}; python -m semantic.run_visualizer ${path} ${SEMANTIC_SATNERF_EVAL_DP} --device=${gpu5} --device_req_free=False --split=train --save_png=True" C-m

# start from first window
tmux select-window -t $session:0