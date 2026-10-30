"""V7: 2052 tagged router values followed by four selection-ready pairs."""

HC_TASKS = 160
ROUTER_TASKS = 33
SELECT_FIRST = 33
SELECT_ROWS = 4
TASK_PUBLISHER = 37
CLEAR_FIRST = 38
CLEAR_TASKS = 10
ROUTER_VALUE_OFFSET = 0
ROUTER_VALUES = 4 * 513
SELECT_READY_OFFSET = ROUTER_VALUES * 8
FLAGS_BYTES = (ROUTER_VALUES + SELECT_ROWS) * 8
# Bank zero belongs to CTA-leader selection waits. Router-value waits use
# one further bank per lane in the 512-thread selector CTA, debug only.
DIAG_THREAD_BANKS = 1 + 512


def producer(region, index):
    if region == "router_values" and 0 <= index < ROUTER_VALUES:
        return (index % 513) // 16
    if region == "selection" and 0 <= index < SELECT_ROWS:
        return SELECT_FIRST + index
    raise ValueError((region, index))


def dependencies(task):
    if SELECT_FIRST <= task < SELECT_FIRST + SELECT_ROWS:
        row = task - SELECT_FIRST
        return tuple(("router_values", row * 513 + i) for i in range(513))
    if task == TASK_PUBLISHER:
        return tuple(("selection", i) for i in range(SELECT_ROWS))
    return ()
