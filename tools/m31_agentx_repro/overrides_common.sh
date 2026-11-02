# Overrides for every local run (sourced by run_local_m31.sh after the CI exports).
#
# The cells name the model by repo id and load it through the CI HF cache
# (${ATOMESH_MODEL_ROOT}/.huggingface/hub, owned by the CI user, read-only for
# me). Load the very snapshot CI loads, by path: revision 0a00e0e3 (the Hub's
# latest). /share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4 is the older
# 1442d09d: same weights, different chat_template.jinja.
M31_SNAPSHOT=/share_nfs/models/.huggingface/hub/models--amd--MiniMax-M3.1-Flash-Quark-MXFP4/snapshots/0a00e0e39fc00499b23efccc18a8ea63fce5c44c
export MODEL_PATH="${M31_SNAPSHOT}"
export DRAFT_MODEL_PATH="${M31_SNAPSHOT}/dspark"
# Only the AgentX dataset (1.85 GB) still goes through HF_HUB_CACHE, which must
# be writable: the container's /tmp.
export ATOMESH_ENV_HF_HUB_CACHE=/tmp/hf-hub
export ATOMESH_PREFILL_ENV_HF_HUB_CACHE=/tmp/hf-hub
export ATOMESH_DECODE_ENV_HF_HUB_CACHE=/tmp/hf-hub
# The launcher's fadvise walk of the whole shared model root (NFS, every model of
# every team) takes ~4 minutes; the privileged drop_caches already emptied the
# page cache, so only the M3.1 checkpoint (read again by the loaders) matters.
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS="${M31_SNAPSHOT}"
