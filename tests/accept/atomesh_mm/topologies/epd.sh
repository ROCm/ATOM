# E/P/D: the PD deployment of topologies/pd.sh plus an encoder-only ATOM
# server (ENCODE_NODE/GPUS, ENCODE_ARGS) that ATOMesh sends each image
# request to before P; P reads the image embeddings from E over Mooncake on
# the HCAs of hca/<node>.map, as it moves KV to D. Checked against pd.
#
# Beside each role's HCA byte counters, the JSON line carries how many image
# requests reached P (image_requests), how many of them E answered
# (encode_served), how many went to P without E's embeddings
# (encode_fallbacks), and how many reads of E's embeddings P's ranks gave up
# for their own vision tower (read_fallbacks, one per rank and request). The
# run fails when E answered none, when every read P made fell back, or when
# E's HCAs sent no RDMA bytes: P then read the embeddings off RDMA, or not at
# all. It also fails as PD's does when the KV moved off RDMA.

source "$HERE/topologies/pd.sh"
REF_TOPOLOGY=pd
HCA_ROLES=(encode prefill decode)

launch_encode() {
  local ATOM_ARGS=("${ENCODE_ARGS[@]}")
  launch_rdma encode "$ENCODE_PORT" "$ENCODE_ENGINE_PORT"
}

topology_up() {
  launch_encode
  launch_pd prefill kv_producer "$PREFILL_PORT" "$PREFILL_ENGINE_PORT"
  launch_pd decode kv_consumer "$DECODE_PORT" "$DECODE_ENGINE_PORT"
  launch_mesh mesh "$MESH_PORT" --backend atom --pd-disaggregation --policy cache_aware \
    --request-timeout-secs 14400 --encode "$(role_url encode "$ENCODE_PORT")" \
    --prefill "$(role_url prefill "$PREFILL_PORT")" --decode "$(role_url decode "$DECODE_PORT")"
  ENDPOINT=$(role_url mesh "$MESH_PORT")
}

# log_lines <role> <extended regex>: how many lines of the role's log in
# RUN_DIR match, colour codes stripped, counted on the role's node: another
# node's view of the shared log can lag behind its writer.
log_lines() {
  local cmd
  cmd="f=$(printf %q "$RUN_DIR/$1.log")
    [[ -r \$f ]] || { echo \"cannot read \$f\" >&2; exit 2; }
    sed 's/\x1b\[[0-9;]*m//g' \"\$f\" | grep -cE $(printf %q "$2") || ((\$? == 1))"
  on_node "$(role_var "$1" NODE)" "$cmd"
}

# topology_findings <counter deltas>: the counts above and the run's failures.
topology_findings() {
  local ranks image_requests served read_fallbacks kv
  IFS=, read -ra ranks <<<"$PREFILL_GPUS"
  # ATOMesh logs a routing decision per pool and request, with the number of
  # media in the request.
  image_requests=$(log_lines mesh 'routing decision.*pool="prefill".*media_digests=[1-9]')
  served=$(log_lines encode '"POST /v1/chat/completions HTTP/1.1" 200')
  read_fallbacks=$(log_lines prefill 'are unavailable; running the vision tower')
  kv=$(kv_failures "$1")
  python3 - "$1" "$kv" "${#ranks[@]}" "$image_requests" "$served" "$read_fallbacks" <<'EOF'
import json
import sys

deltas = json.loads(sys.argv[1])
failures = json.loads(sys.argv[2])
ranks, image_requests, served, read_fallbacks = map(int, sys.argv[3:])
if not served:
    failures.append("E served no request")
elif read_fallbacks >= served * ranks:
    failures.append(f"every read of E's embeddings fell back ({read_fallbacks})")
if served and not deltas["encode_hca_tx_bytes"]:
    failures.append("E's HCAs sent no RDMA bytes: the embeddings left RDMA")
print(
    json.dumps(
        {
            "image_requests": image_requests,
            "encode_served": served,
            "encode_fallbacks": image_requests - served,
            "read_fallbacks": read_fallbacks,
            "failures": failures,
        }
    )
)
EOF
}
