"""Use ATOM's existing ReplaySSM switch for the RTP hybrid implementation."""
import torch
from atom.utils import envs


def replay_mode():
    override = envs.ATOM_ENABLE_REPLAYSSM
    enabled = bool(torch.version.hip) if override is None else override
    return "1" if enabled else "0"


def flydsl_mtp_enabled():
    # ReplaySSM selects the hybrid algorithm, which also uses FlyDSL MTP.
    return bool(torch.version.hip)
