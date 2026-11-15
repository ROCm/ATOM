"""Construct the flat probability records used by engine/frontend regressions."""


def token_logprob(value, rank, top=()):
    return (value, rank) + tuple(item for pair in top for item in pair)
