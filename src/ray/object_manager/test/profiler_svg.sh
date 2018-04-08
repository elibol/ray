#!/usr/bin/env bash

CORE_DIR="${CORE_DIR:-/home/ubuntu/ray/python/ray/core}"
om_exec="$CORE_DIR/src/ray/object_manager/object_manager_benchmark_tool"

google-pprof -svg $om_exec ./$1 > ./$1.svg

# If you realize the call graph is too large, use -focus=<some function> to zoom
# into subtrees.
# google-pprof -focus=epoll_wait -svg $om_exec /tmp/pprof.out > /tmp/pprof.svg