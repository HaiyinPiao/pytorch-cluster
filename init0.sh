#!/bin/bash
# export MASTER_ADDR=192.168.1.101
# export MASTER_ADDR=192.168.1.100
# export MASTER_PORT=29500
# export GLOO_SOCKET_IFNAME=eno1

# kill_children() {
#   for PID in ${PIDS[*]}; do
#     kill -TERM $PID
#   done
# }

NODE=0 #Change this for each machine indexed with 0, 1, 2 ... N
RANKS_PER_NODE=1 #For P3.16x machine
WORLD_NODE=2

# for i in $(seq 0 7); do
#   LOCAL_RANK=$i
#   RANK=$((RANKS_PER_NODE * NODE + LOCAL_RANK))
#   python ./dist.py  \
#        --world_size $((WORLD_NODE * RANKS_PER_NODE)) \
#        --ranks_per_node 8 \
#        --rank $RANK
#   PIDS[$LOCAL_RANK]=$!
# done

python ./dist.py  \
     --world_size $((WORLD_NODE * RANKS_PER_NODE)) \
     --node $NODE \
     --ranks_per_node $RANKS_PER_NODE \

# PIDS[$LOCAL_RANK]=$!

# trap kill_children SIGTERM SIGINT

# for PID in ${PIDS[*]}; do
#   wait $PID
# done
