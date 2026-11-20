# Direct: one ATOM server, no ATOMesh. The reference other topologies are
# checked against.

topology_up() {
  launch_atom atom "$ATOM_PORT" "$ATOM_ENGINE_PORT"
  ENDPOINT=$(role_url atom "$ATOM_PORT")
}
