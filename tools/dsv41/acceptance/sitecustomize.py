"""One explicit R13 serving bootstrap, also inherited by spawned workers.

Enabled only by the DSV41 recipe. The installation order is the measured
order; no previous experiment directory or chained sitecustomize is loaded.
"""

import os


def install():
    if os.environ.get("DSV41_CONSOLIDATED_ENTRY") != "1":
        return
    server = os.environ.get("DSV41_REBASE_SERVICE_SERVER") == "1"
    owner = os.environ["DSV41_OWNER_SERVING"] == "1"
    if server:
        from pathlib import Path

        import atom

        source = Path(os.environ["DSV41_REBASE_ATOM_SOURCE"]).resolve()
        if not Path(atom.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Wrong DSV41 subject imported")
        if owner:
            from atom.models.deepseek_v41.owner_service import install as execution
        else:
            from origin_native_service_accounting import install as execution
        execution()
        from owner_initialization_progress import install as progress
        from rebase_graph_audit_r6 import install as audit
        from serving_commit_accounting import install as commits
        from step12_model_preload_r5 import install as preload

        progress()
        commits()
        audit()
        preload()
        if owner:
            from step12_grouped_route import install as grouped

            grouped()
        from step12_loader_phase_timing import install as loader_timing
        from step12_model_preload_r9 import install as index_preload
        from step12_prefill_ffn_route import install as prefill

        index_preload()
        prefill()
        loader_timing()

    from run_rebase_graph_coverage_r6 import SERVER_COMMAND

    ep = os.environ["DSV41_STEP12_OWNER_EP4"] == "1"
    if ep != owner or "--enable-expert-parallel" in SERVER_COMMAND:
        raise RuntimeError("Invalid or duplicate DSV41 deployment bootstrap")
    if ep:
        SERVER_COMMAND.append("--enable-expert-parallel")
    if server:
        from step12_owner_ep_service_audit import install as ep_audit
        from step12_placement_service_audit import install as placement_audit

        ep_audit()
        placement_audit()
    else:
        from step12_ep_client_environment import install_client_environment

        install_client_environment()


try:
    install()
except BaseException:  # noqa: BLE001 - never run a partly installed recipe
    import traceback

    traceback.print_exc()
    os._exit(78)
