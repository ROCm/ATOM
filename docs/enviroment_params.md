# ATOM Enviroment Parameters Usage

This doc introduce the usage of envoriment parameters in ATOM.

## Common
- **ATOM_ENFORCE_EAGER**
  
  **Default:** 0

  **Description:** If set to 1, eager mode will be used instead of the CUDA Graph for kernel launch.

- **ATOM_USE_TRITON_GEMM**

  **Default:** 0

  **Description:** If set 1, AITER Triton FP4 weight preshuffled gemm will be selected, otherwise AITER ASM FP4 weight preshuffled gemm will be used.



## Data Parallelism

| Column | Description | Default | 
|--------|-------------|---------|
| **ATOM_DP_RANK** | The rank id for the current process | 0 |
| **ATOM_DP_RANK_LOCAL** |  | 0 |
| **ATOM_DP_SIZE** | The total size of the DP rank | 1 |
| **ATOM_DP_MASTER_IP** | Master ip address of all the DP ranks | 127.0.0.1 |
| **ATOM_DP_MASTER_PORT** | Master port of all the DP ranks | 1 |


## Fusion Pass 

#### 1. TP AllReduce Fusion
- **ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION**
  
  **Default:** 1

  **Description:** If set 1, fuse allreduce with rmsnorm in TP mode.


#### 2. Deepseek-style
- **ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION**
  
  **Default:** 1

  **Description:** If set 1, fuse rmsnorm with quantization.

- **ATOM_ENABLE_DS_QKNORM_QUANT_FUSION**
  
  **Default:** 1

  **Description:** If set 1, fuse the qk_norm with quantization in MLA attention module. 

- **ATOM_USE_TRITON_MXFP4_BMM**
  
  **Default:** 0

  **Description:** If set 1, use FP4 BMM in MLA attention module. 


#### 3. Llama-style
- **ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT**
  
  **Default:** 1

  **Description:** If set 1, use the triton kernel to fuse rmsnorm with quantization.

- **ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT**
  
  **Default:** 1

  **Description:** If set 1, use the triton kernel to fuse silu and mul together with quantization in MLP module. 