#!/usr/bin/env bash

source config.sh

NODE_IP="127.0.0.1"
REDIS_IP="127.0.0.1"
REDIS_PORT="6379"
PROFILE_OUT="$PROFILE_DIR/om_send_pprof.out"
SVG_OUT="$PROFILE_DIR/om_send_pprof.svg"

MODE="send"
# MODE="bidirectional"

# echo " $NODE_IP $REDIS_IP $REDIS_PORT $STORE_EXEC $MODE $OBJECT_SIZE $NUM_OBJECTS $NUM_TRIALS $MAX_SENDS $MAX_RECEIVES $OBJECT_CHUNK_SIZE $STORE_GIGABYTE_MEMORY $SKIP_K"
# gdb $OM_EXEC
$OM_EXEC  $NODE_IP $REDIS_IP $REDIS_PORT $STORE_EXEC $MODE $OBJECT_SIZE $NUM_OBJECTS $NUM_TRIALS $MAX_SENDS $MAX_RECEIVES $OBJECT_CHUNK_SIZE $STORE_GIGABYTE_MEMORY $SKIP_K
# LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=$PROFILE_OUT $OM_EXEC $NODE_IP $REDIS_IP $REDIS_PORT $STORE_EXEC $MODE $OBJECT_SIZE $NUM_OBJECTS $NUM_TRIALS $MAX_SENDS $MAX_RECEIVES $OBJECT_CHUNK_SIZE $STORE_GIGABYTE_MEMORY $SKIP_K
# google-pprof -svg $OM_EXEC $PROFILE_OUT > $SVG_OUT
