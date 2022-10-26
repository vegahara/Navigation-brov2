#!/bin/sh

tmux new-session -d -s navigation 
tmux send-keys 'source install/setup.bash' 'C-m'
tmux rename-window 'Navigation'
tmux select-window -t navigation:0
tmux split-window -h
tmux send-keys 'source install/setup.bash' 'C-m'
tmux split-window -v -t 0 
tmux send-keys 'source install/setup.bash' 'C-m'
tmux split-window -v -t 1 
tmux send-keys 'source install/setup.bash' 'C-m'
tmux -2 attach-session -t navigation