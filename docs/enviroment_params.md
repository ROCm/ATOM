# ATOM Enviroment Parameters Usage

This doc introduce the usage of envoriment parameters in ATOM.

## Common
| Column | Description | Default | 
|--------|-------------|---------|
| ATOM_ENFORCE_EAGER | Use eager mode instead of the CUDA Graph for kernel launch | 0 |

## Computation
| Column | Description | Default | 
|--------|-------------|---------|
| ATOM_USE_TRITON_GEMM | Use AITER Triton Gemm for computation | 0 |

## Parallelism

### Data Parallel
| Column | Description | Default | 
|--------|-------------|---------|
| ATOM_DP_RANK | The rank id for the current process | 0 |
| ATOM_DP_RANK_LOCAL |  | 0 |
| ATOM_DP_SIZE | The total size of the DP rank | 1 |
| ATOM_DP_MASTER_IP | Master ip address of all the DP ranks | 127.0.0.1 |
| ATOM_DP_MASTER_PORT | Master port of all the DP ranks | 1 |

### Tensor Parallel
| Column | Description | Default | 
|--------|-------------|---------|
| ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION | Fuse allreduce (TP) with rmsnorm | 1 |

## Fusion Pass 

### Deepseek-style

  | Column | Description | Default | 
  |--------|-------------|---------|
  | ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION | fuse rmsnorm with quantization | on |
  | ATOM_ENABLE_DS_QKNORM_QUANT_FUSION | fuse the qk_norm with quantization | on |
  | ATOM_USE_TRITON_MXFP4_BMM | Use FP4 bmm for DS Prefill computation | off |

### Llama-style

  | Column | Description | Default | 
  |--------|-------------|---------|
  | ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT | use the triton kernel to fuse rmsnorm with quantization | on |
  | ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT | use the triton kernel to fuse silu, mul with quantization in MLP module | on |
