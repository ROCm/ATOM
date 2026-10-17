"""Reject a CPU offload pool that exceeds ordinary physical RAM on this node."""

from pathlib import Path


def inspect_host_capacity(config, role, meminfo=None):
    # The pinned C48 SimpleCPU implementation divides this node-wide budget
    # among the eight TP workers. Explicit HugeTLB reservations are separate
    # from the ordinary torch.zeros allocations used by its CPU backend.
    transfer = config["roles"][role]["args"]["kv-transfer-config"]
    connectors = transfer.get("kv_connector_extra_config", {}).get(
        "connectors", [transfer]
    )
    budgets = [
        int(c["kv_connector_extra_config"]["cpu_bytes_to_use"])
        for c in connectors
        if c["kv_connector"] == "SimpleCPUOffloadConnector"
    ]
    requested = sum(budgets)
    if meminfo is None:
        meminfo = Path("/proc/meminfo").read_text()
    values = {}
    for line in meminfo.splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        values[key] = int(fields[0]) * (1024 if fields[1:] == ["kB"] else 1)
    total = values["MemTotal"]
    reserved = values.get("Hugetlb", 0)
    ordinary = total - reserved
    return {
        "role": role,
        "requested_cpu_cache_bytes_node": requested,
        "mem_total_bytes": total,
        "hugetlb_reserved_bytes": reserved,
        "huge_pages_total": values.get("HugePages_Total"),
        "huge_pages_free": values.get("HugePages_Free"),
        "mem_available_bytes": values["MemAvailable"],
        "ordinary_physical_bytes": ordinary,
        "impossible_capacity": requested > ordinary,
        "scope": "Hard physical-capacity rejection only; passing is not a guarantee of pinnable memory or headroom.",
    }
