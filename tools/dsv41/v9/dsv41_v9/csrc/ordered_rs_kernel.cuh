// Local diagnostic candidate: exact per-segment BF16 order, same start/end sync.
template<typename T>
__global__ void __launch_bounds__(512, 1) dsv41_rs_bf16_ordered(
    RankData* dp, RankSignals sg, Signal* self_sg, T* result,
    int rank, int range, const uint8_t* order_codes)
{
    constexpr int pack = 16 / sizeof(T);
    using P = typename opus::vector_t<T, pack>;
    const P* ptrs[4];
    #pragma unroll
    for(int r=0;r<4;r++) ptrs[r]=reinterpret_cast<const P*>(dp->ptrs[r]);
    start_sync<4>(sg,self_sg,rank);
    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=gridDim.x*blockDim.x;
    for(int idx=tid;idx<range;idx+=stride){
        const uint8_t code=order_codes[(idx*pack)/128];
        const int load_index=rank*range+idx;
        P value=ptrs[code&3][load_index];
        #pragma unroll
        for(int j=1;j<4;j++) {
            P other=ptrs[(code>>(2*j))&3][load_index];
            packed_assign_add<T,pack>(value,other);
        }
        reinterpret_cast<P*>(result)[idx]=value;
    }
    end_sync<4,true>(sg,self_sg,rank);
}
