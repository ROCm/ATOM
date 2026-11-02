#!/bin/bash
# down.sh: stop this user's server -- only the processes inside $CONTAINER, which is not --pid=host -- and wait until
# they and their GPU contexts are gone.
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env
docker exec "$CONTAINER" pkill -INT -f '[a]tom.entrypoints.openai_server' 2> /dev/null
for _ in $(seq 1 24); do
  n=$(docker top "$CONTAINER" -eo pid,args 2> /dev/null | grep -cE 'openai_server|ATOM::')
  [ "${n:-0}" = 0 ] && break
  sleep 5
done
docker exec "$CONTAINER" pkill -KILL -f '[a]tom.entrypoints.openai_server|ATOM::' 2> /dev/null
sleep 10
echo "server in $CONTAINER stopped"
