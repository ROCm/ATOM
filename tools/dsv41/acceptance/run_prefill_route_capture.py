"""One bounded untimed C256 diagnostic using the accepted serving contract."""


def bounded_log(pipe, path, limit):
    written = 0
    with path.open("xb") as output:
        while block := pipe.read1(8192):
            retained = block[: max(0, limit - written)]
            output.write(retained)
            output.flush()
            written += len(retained)
    pipe.close()
