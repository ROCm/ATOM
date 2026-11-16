# Single: one ATOM server behind ATOMesh, cache_aware routing; checked
# against direct. A server launched with USE_ATOMESH_ENTRYPOINTS=1 is ATOMesh
# itself, and serves the requests alone.

REF_TOPOLOGY=direct

topology_up() {
  launch_atom atom "$ATOM_PORT" "$ATOM_ENGINE_PORT"
  ENDPOINT=$(role_url atom "$ATOM_PORT")
  if standalone; then return 0; fi
  launch_mesh mesh "$MESH_PORT" --backend atom --policy cache_aware \
    --request-timeout-secs 14400 --worker-urls "$ENDPOINT"
  ENDPOINT=$(role_url mesh "$MESH_PORT")
}
