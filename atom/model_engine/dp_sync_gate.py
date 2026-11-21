"""Which DP (DP > 1) busy-loop rounds run the cross-DP state all_reduce."""

# Largest ATOM_DP_SYNC_INTERVAL: every rank still runs up to N-1 dummy rounds
# after the group drains, and commands, abort and shutdown wait as long.
MAX_DP_SYNC_INTERVAL = 16


class DPSyncGate:
    """Skip the DP state all_reduce for the rounds after a sync that saw work.

    Every rank feeds it the same all_reduce results and runs the same number of
    busy rounds between syncs, so every rank syncs on the same round.
    Ranks with no local work still complete a paired dummy step. All ranks
    must use the same interval, 2 to ``MAX_DP_SYNC_INTERVAL`` (interval 1, every
    round, needs no gate: `DPEngineCoreProc.busy_loop`). The rounds after a
    sync that saw work run paired, so once the group drains every rank still
    runs up to ``interval - 1`` dummy rounds before the next sync sees it idle;
    utility commands and shutdown wait for that sync too.
    """

    def __init__(self, interval: int) -> None:
        self.interval = interval
        self._busy_rounds = 0
        self._last_sync_busy = False

    def should_sync(self) -> bool:
        return not self._last_sync_busy or self._busy_rounds % self.interval == 0

    def synced(self, has_unfinished: bool, shutdown: bool, held: bool) -> None:
        self._last_sync_busy = has_unfinished and not shutdown and not held
        self._busy_rounds = 0

    def busy_round_done(self) -> None:
        self._busy_rounds += 1
