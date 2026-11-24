# ARM B0 = arm B with mono OFF (shippable fallback if the mono gate fails). Everything else identical to env_armB.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armB.sh"
export ATOM_MONO_ENABLE=0
# ARM stays B so the node-local flydsl cache of arm B is reused (numerics unaffected)
