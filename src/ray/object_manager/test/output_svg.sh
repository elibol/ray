#!/usr/bin/env bash

source config.sh

google-pprof -svg $OM_EXEC $1 > $1.svg

# If you realize the call graph is too large, use -focus=<some function> to zoom
# into subtrees.
# google-pprof -focus=epoll_wait -svg $om_exec /tmp/pprof.out > /tmp/pprof.svg
