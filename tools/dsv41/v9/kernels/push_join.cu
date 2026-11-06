#include "native_include/custom_all_reduce.cuh"
#include <cstdio>
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
namespace dsv41_push {
using T=opus::bf16_t;
using P=opus::vector_t<T,8>;
using A=opus::vector_t<float,8>;
struct Args {
    const T* routed;
    const T* shared;
    T* out;
    const aiter::RankData* scratch;
    const uint8_t* codes;
    int rows;
    int range;
};
template<int Routes>
__global__ __launch_bounds__(512,1)
void join_rs(Args a,aiter::RankSignals signals,aiter::Signal* self,int rank) {
    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=gridDim.x*blockDim.x;
    aiter::start_sync<4>(signals,self,rank);
    // Each block produces and later consumes the identical striped chunk.
    for(int idx=tid;idx<a.range;idx+=stride) {
        #pragma unroll
        for(int dest=0;dest<4;++dest) {
            const int global=dest*a.range*8+idx*8;
            const int token=global/5120;
            P combined{};
            if(token<a.rows) {
                P rounded;
                if constexpr(Routes==6) {
                    const int col=global-token*5120;
                    const int base=token*6*5120+col;
                    A acc{};
                    #pragma unroll
                    for(int k=0;k<6;++k) {
                        P v=*reinterpret_cast<const P*>(a.routed+base+k*5120);
                        acc=acc+aiter::upcast(v);
                    }
                    rounded=aiter::downcast<P>(acc);
                } else {
                    rounded=*reinterpret_cast<const P*>(a.routed+global);
                }
                P shared=*reinterpret_cast<const P*>(a.shared+global);
                combined=aiter::downcast<P>(aiter::upcast(rounded)+aiter::upcast(shared));
            }
            auto* dst=reinterpret_cast<P*>(const_cast<void*>(a.scratch->ptrs[dest]));
            dst[rank*a.range+idx]=combined;
        }
    }
    // This is not final: remote scratch writes must be visible to consumers.
    // Every producer thread must finish its peer VMEM before the signaling
    // wave publishes completion. A release executed only by the four signal
    // threads is insufficient for outstanding stores issued by other waves.
    __threadfence_system();
    aiter::end_sync<4,false>(signals,self,rank);
    const auto* local=reinterpret_cast<const P*>(a.scratch->ptrs[rank]);
    for(int idx=tid;idx<a.range;idx+=stride) {
        const uint8_t code=a.codes[(idx*8)/128];
        P value=local[(code&3)*a.range+idx];
        #pragma unroll
        for(int j=1;j<4;++j) {
            P other=local[((code>>(2*j))&3)*a.range+idx];
            aiter::packed_assign_add<T,8>(value,other);
        }
        reinterpret_cast<P*>(a.out)[idx]=value;
    }
    aiter::end_sync<4,true>(signals,self,rank);
}
}
static thread_local char error[1024];
extern "C" const char* dsv41_push_last_error(){return error;}
extern "C" int dsv41_push_join_rs(int64_t fa_ptr,int64_t routed,int64_t shared,
    int64_t output,int64_t scratch,int64_t codes,int64_t rows,int64_t padded,
    int64_t routes,int64_t stream_ptr) {
    try {
        if(rows<4096||padded>16384||padded%4||padded<rows||padded-rows>3||
           (routes!=1&&routes!=6)) throw std::runtime_error("Pushjoin invalid shape");
        auto* fa=reinterpret_cast<aiter::CustomAllreduce*>(fa_ptr);
        if(fa->world_size_!=4||!fa->full_nvlink_) throw std::runtime_error("Pushjoin requires TP4 full connectivity");
        auto stream=reinterpret_cast<hipStream_t>(stream_ptr);
        dsv41_push::Args a{reinterpret_cast<const dsv41_push::T*>(routed),
            reinterpret_cast<const dsv41_push::T*>(shared),reinterpret_cast<dsv41_push::T*>(output),
            fa->get_buffer_RD(stream,reinterpret_cast<void*>(scratch)),
            reinterpret_cast<const uint8_t*>(codes),static_cast<int>(rows),static_cast<int>(padded*5120/(4*8))};
        if(routes==6) dsv41_push::join_rs<6><<<80,512,0,stream>>>(a,fa->sg_,fa->self_sg_,fa->rank_);
        else dsv41_push::join_rs<1><<<80,512,0,stream>>>(a,fa->sg_,fa->self_sg_,fa->rank_);
        HIP_CALL(hipGetLastError());return 0;
    } catch(const std::exception& e) {snprintf(error,sizeof(error),"%s",e.what());return 1;}
}
