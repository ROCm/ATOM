# Sourced last in every rank env: shift every launcher port of the combined phase by +1000 (with
# p2d/patches/combined_port_offset.patch): another user's CPU-only simulation on g27 listens on 29474-29601.
export ATOMESH_COMBINED_PORT_OFFSET=1000
