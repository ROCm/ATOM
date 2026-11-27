#!/bin/bash
# fab_down.sh: stop the router (on P) and both servers. Always both: a restarted D on the same host:port would be
# written through P's cached fabric session.
cd "$(dirname "$0")" && . ./common.sh
on "$P_HOST" "docker exec $CTR pkill -INT -f '[a]tomesh launch' 2>/dev/null; sleep 3; docker exec $CTR pkill -KILL -f '[a]tomesh launch' 2>/dev/null; echo router stopped" 60
on "$P_HOST" "cd $KIT_REMOTE && CONTAINER=$CTR ./down.sh" 300 & a=$!
on "$D_HOST" "cd $KIT_REMOTE && CONTAINER=$CTR ./down.sh" 300 & b=$!
wait $a $b
