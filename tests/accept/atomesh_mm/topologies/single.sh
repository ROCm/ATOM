# Single: one ATOM server behind ATOMesh, cache_aware routing; checked
# against direct.

REF_TOPOLOGY=direct

topology_up() {
  launch_atom atom "$ATOM_PORT" "$ATOM_ENGINE_PORT"
  launch_mesh mesh "$MESH_PORT" --backend atom --policy cache_aware \
    --request-timeout-secs 14400 --worker-urls "$(role_url atom "$ATOM_PORT")"
  ENDPOINT=$(role_url mesh "$MESH_PORT")
}
