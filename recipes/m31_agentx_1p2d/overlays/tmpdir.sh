# Sourced in every rank env before pd_slurm_job.sh. The campaign image ships a root-owned /tmp/aiter_configs (left by its
# root build), and aiter merges its tuned CSVs into tempfile.gettempdir()/aiter_configs, which the launcher's non-root
# containers cannot write. /dev/shm exists in every container (the host's tmpfs under --ipc host) and is world-writable.
export ATOMESH_ENV_TMPDIR=/dev/shm
