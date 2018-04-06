CORE_DIR="/home/elibol/dev/ray/python/ray/core"

node_ip_address="127.0.0.1"
redis_address="127.0.0.1"
redis_port="6379"
store_executable="$CORE_DIR/src/plasma/plasma_store"
om_exec="$CORE_DIR/src/ray/object_manager/om_multinode"

mode="send"
object_size=1000000000
num_objects=3

# echo "$node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects"
# gdb $om_exec 
$om_exec $node_ip_address $redis_address $redis_port $store_executable $mode $object_size $num_objects
