#!/usr/bin/env bash

CORE_DIR="${CORE_DIR:-/home/ubuntu/ray/python/ray/core}"
STORE_EXEC="$CORE_DIR/src/plasma/plasma_store"
OM_EXEC="$CORE_DIR/src/ray/object_manager/object_manager_benchmark_tool"
PROFILE_DIR="$CORE_DIR/../../../src/ray/object_manager/test/profile"
if [ ! -d $PROFILE_DIR ]; then
  mkdir -p $PROFILE_DIR;
fi


MAX_THREADS=4
MAX_SENDS=4
MAX_RECEIVES=4
# note that the sender plasma store will need at lease
# object_size*num_objects*num_trials memory to run the experiment.
STORE_GIGABYTE_MEMORY=8
      OBJECT_SIZE=10000000
OBJECT_CHUNK_SIZE=1000000
NUM_OBJECTS=1
NUM_TRIALS=10
