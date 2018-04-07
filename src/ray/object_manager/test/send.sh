#!/usr/bin/env bash

CORE_DIR="${CORE_DIR:-/home/ubuntu/ray/python/ray/core}"

node_ip_address="127.0.0.1"
redis_address="127.0.0.1"
redis_port="6379"
store_executable="$CORE_DIR/src/plasma/plasma_store"
om_exec="$CORE_DIR/src/ray/object_manager/object_manager_benchmark_tool"

mode="send"
# note that the sender plasma store will need at lease
# object_size*num_objects*num_trials memory to run the experiment.
object_size=100000000
num_objects=2
num_trials=10

num_threads=4
max_sends=4
max_receives=4

# echo "$node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects $num_trials $num_threads $max_sends $max_receives"
# gdb $om_exec 
$om_exec $node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects $num_trials $num_threads $max_sends $max_receives
