"""C1 HC/router/task phase ownership; G1/G2 follow on the same stream."""

HC_TASKS = 160
ROUTER_TASKS = 33
SELECT_FIRST = 33
SELECT_ROWS = 4
TASK_PUBLISHER = 37
CLEAR_FIRST = 38
CLEAR_TASKS = 10
ROUTER_READY_OFFSET = 0
SELECT_READY_OFFSET = ROUTER_TASKS * 8
FLAGS_BYTES = (ROUTER_TASKS + SELECT_ROWS) * 8


def producer(region, index):
    if region == "router" and 0 <= index < ROUTER_TASKS:
        return index
    if region == "selection" and 0 <= index < SELECT_ROWS:
        return SELECT_FIRST + index
    raise ValueError((region, index))


def dependencies(task):
    if SELECT_FIRST <= task < SELECT_FIRST + SELECT_ROWS:
        return tuple(("router", i) for i in range(ROUTER_TASKS))
    if task == TASK_PUBLISHER:
        return tuple(("selection", i) for i in range(SELECT_ROWS))
    return ()
