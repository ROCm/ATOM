from atom.config import Config
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.spec_decode.eagle_proposer import EagleProposer


def build_drafter(config: Config, device, runner) -> Drafter:
    """Construct the speculative drafter for this config.

    Block-parallel DSpark vs serial EAGLE/MTP is the only flavor branch; every
    downstream call-site depends on the ``Drafter`` contract, not the concrete
    type.
    """
    return drafter_class(config)(config, device, runner)


def drafter_class(config: Config) -> type[Drafter]:
    """The drafter class this config speculates with, on every PP stage."""
    if config.speculative_config.use_dspark():
        return DSparkProposer
    return EagleProposer
