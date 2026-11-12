# SPDX-License-Identifier: MIT
"""Common scheduler lookup contract, including non-driving clients."""


class LookupClient:
    token_database = None
    driving = False

    def pump(self):
        """Advance owned lookups without waiting; passive clients do nothing."""

    def pending_ids(self):
        return ()

    def is_pending(self, lookup_id):
        return False

    def answered(self, lookup_id):
        return lookup_id in self.pending_ids() and not self.is_pending(lookup_id)

    def query_started(self, lookup_id):
        return None

    def close(self):
        """Release resources owned by this lookup client, if any."""


class LookupClientAdapter(LookupClient):
    """Bind optional external LMCache methods once at the integration boundary."""

    def __init__(self, client):
        self.client = client
        self.token_database = getattr(client, "token_database", None)
        self.driving = bool(getattr(client, "driving", False))
        self._methods = {}
        for name in (
            "answered",
            "is_pending",
            "pending_ids",
            "pump",
            "query_started",
            "close",
        ):
            method = getattr(client, name, None)
            self._methods[name] = method if callable(method) else None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def answered(self, lookup_id):
        method = self._methods["answered"]
        return super().answered(lookup_id) if method is None else method(lookup_id)

    def is_pending(self, lookup_id):
        method = self._methods["is_pending"]
        return False if method is None else method(lookup_id)

    def pending_ids(self):
        method = self._methods["pending_ids"]
        return () if method is None else method()

    def pump(self):
        method = self._methods["pump"]
        if method is not None:
            method()

    def query_started(self, lookup_id):
        method = self._methods["query_started"]
        return None if method is None else method(lookup_id)

    def close(self):
        method = self._methods["close"]
        if method is not None:
            method()


def as_lookup_client(client):
    return (
        client
        if client is None or isinstance(client, LookupClient)
        else LookupClientAdapter(client)
    )
