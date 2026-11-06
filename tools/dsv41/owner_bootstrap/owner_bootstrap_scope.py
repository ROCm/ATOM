"""Keep the optional owner bootstrap out of ROCm compiler helper processes."""


def service_process(original_argv):
    args = tuple(original_argv)
    for i, arg in enumerate(args[:-1]):
        if arg == "-m" and args[i + 1] == "atom.entrypoints.openai_server":
            return True
        if arg == "-c" and args[i + 1].startswith(
            "from multiprocessing.spawn import spawn_main;"
        ):
            return True
    return False
