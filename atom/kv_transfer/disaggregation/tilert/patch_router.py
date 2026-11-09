# SPDX-License-Identifier: MIT
"""Create an experimental ATOM HTTP bridge from the installed TileRT0.1.7 router."""

import argparse
import inspect
from pathlib import Path


def main():
    from tilert.pd_vllm import pd_router

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = inspect.getsource(pd_router)
    start = source.index("def first_token_from_logprobs(")
    end = source.index("\ndef _token_id_of(", start)
    source = (
        source[:start]
        + "def first_token_from_logprobs(resp: dict, is_chat: bool) -> int:\n"
        + "    return int(resp['kv_transfer_params']['first_token_id'])\n"
        + source[end:]
    )
    old = "    return prefill_body"
    if source.count(old) != 1 or source.count("derive_rid(prefill['id'])") != 2:
        raise RuntimeError("Unexpected TileRT router version; inspect before patching")
    source = source.replace(
        old,
        "    prefill_body.pop('logprobs', None)\n"
        "    prefill_body.pop('top_logprobs', None)\n"
        "    prefill_body['kv_transfer_params']['do_remote_decode'] = True\n"
        + old,
    ).replace("derive_rid(prefill['id'])", "prefill['kv_transfer_params']['rid']")
    compile(source, str(args.output), "exec")
    with args.output.open("x") as output:
        output.write(source)


if __name__ == "__main__":
    main()
