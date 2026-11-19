"""Which multi-node DP busy-loop rounds run the cross-DP state all_reduce."""


class DPSyncGate:
    """Skip the DP state all_reduce between syncs that found every rank busy.

    Every rank feeds it the same all_reduce results and runs the same number of
    busy rounds between syncs, so every rank syncs on the same round.
    ``interval == 1`` syncs every round.
    """

    def __init__(self, interval: int) -> None:
        self.interval = max(1, int(interval))
        self._busy_rounds = 0
        self._last_sync_busy = False

    def should_sync(self) -> bool:
        return (
            self.interval == 1
            or not self._last_sync_busy
            or self._busy_rounds % self.interval == 0
        )

    def synced(self, has_unfinished: bool, shutdown: bool, held: bool) -> None:
        self._last_sync_busy = has_unfinished and not shutdown and not held
        self._busy_rounds = 0

    def busy_round_done(self) -> None:
        self._busy_rounds += 1
