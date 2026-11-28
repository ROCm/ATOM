# miles + AtomRL plugin

Qwen3-8B Megatron training on miles with AtomRL's Lumen FP8 hooks. miles keeps
its native SGLang rollout by default; `atomrl.plugin.miles.hooks.generate_rollout`
is available when the trainer explicitly replaces rollout with ATOM.

## Hooks

`atomrl.plugin.miles.hooks` exposes the host-defined top-level hooks for model
construction, FP8 reset, rollout, checkpoint callbacks, and rank-zero ATOM
weight reload. `atomrl.plugin.miles.lifecycle` creates and tears down the public
`ATOMReplicaManager`.

The adapter imports miles only when a miles-native rollout result is requested,
so installing AtomRL does not require miles.

## Build

The miles repository has no public default in this example. Supply the reviewed
repository and revision explicitly:

```bash
docker build -f atomrl/examples/plugin/miles/Dockerfile \
  --build-arg MILES_REPOSITORY=<miles repository URL> \
  --build-arg MILES_REF=<reviewed revision> \
  -t atomrl-miles-plugin .
```

## Run

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --ipc=host --network=host --privileged \
  -v /data:/data -e DATA_ROOT=/data \
  atomrl-miles-plugin
```

Set `TRAIN_FP8=1` to enable Lumen training patches and `STEPS=<n>` to change
the smoke length. This example remains `unvalidated` in the unified harness
until a reviewed miles source revision and runnable image are supplied.
