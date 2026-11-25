# slime + AtomRL plugin

AtomRL provides slime-native rollout outputs, Megatron/Lumen hooks, public ATOM
manager lifecycle, and rank-zero checkpoint reload.

The validated source revision is:

```text
https://github.com/THUDM/slime
79aa41a884ab0bf65222c94418f6357d3c31192b
```

Lumen-RL did not provide a slime training example, so this directory contains
an installation/import contract smoke rather than an invented training recipe.
From the ATOM repository root:

```bash
docker build -f atomrl/examples/plugin/Dockerfile.plugin \
  --target slime-test -t atomrl-plugin-test-slime .

docker run --rm atomrl-plugin-test-slime \
  bash atomrl/examples/plugin/slime/run.sh
```

Use the pinned slime documentation when wiring these hooks into a real training
launcher. A training result is not claimed until such a launcher is run.
