#!/usr/bin/env bash

CORE_DIR="${CORE_DIR:-/home/ubuntu/ray/python/ray/core}"

node_ip_address="127.0.0.1"
redis_address="127.0.0.1"
redis_port="6379"
store_executable="$CORE_DIR/src/plasma/plasma_store"

OM_EXEC="$CORE_DIR/src/ray/object_manager/object_manager_benchmark_tool"
PROFILE_DIR="$CORE_DIR/../../../src/ray/object_manager/test/profile"
PROFILE_OUT="$PROFILE_DIR/om_send_pprof.out"

if [ ! -d $PROFILE_DIR ]; then
  mkdir -p $PROFILE_DIR;
fi

mode="send"
object_size=10000000
num_objects=4
num_trials=10

num_threads=4
max_sends=4
max_receives=4

# note that the sender plasma store will need at lease
# object_size*num_objects*num_trials memory to run the experiment.

# echo "$node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects $num_trials $num_threads $max_sends $max_receives"
# gdb $OM_EXEC
# $OM_EXEC $node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects $num_trials $num_threads $max_sends $max_receives
LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=$PROFILE_OUT $OM_EXEC $node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects $num_trials $num_threads $max_sends $max_receives
