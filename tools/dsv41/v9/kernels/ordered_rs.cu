#include "custom_all_reduce_ordered.cuh"
#include <cstdio>
#include <cstring>
static thread_local char last_error[1024];
extern "C" const char* dsv41_rs_last_error(){return last_error;}
extern "C" int dsv41_rs_ordered(int64_t fa_ptr, int64_t inp_ptr,
    int64_t out_ptr, int64_t codes_ptr, int64_t stream_ptr,
    int64_t reg_ptr, int64_t reg_bytes)
{
    try{
        auto stream=reinterpret_cast<hipStream_t>(stream_ptr);
        void* input=reinterpret_cast<void*>(inp_ptr);
        constexpr int64_t bytes=1536LL*5120*2;
        if(reg_ptr){
            if(reg_bytes<bytes) throw std::runtime_error("Input pool too small");
            auto status=hipMemcpyAsync(reinterpret_cast<void*>(reg_ptr),input,bytes,hipMemcpyDeviceToDevice,stream);
            if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
            input=reinterpret_cast<void*>(reg_ptr);
        }
        auto fa=reinterpret_cast<aiter::CustomAllreduce*>(fa_ptr);
        fa->dispatchDsv41RSOrdered(stream,reinterpret_cast<opus::bf16_t*>(input),
            reinterpret_cast<opus::bf16_t*>(out_ptr),reinterpret_cast<const uint8_t*>(codes_ptr));
        auto status=hipGetLastError();
        if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
        return 0;
    }catch(const std::exception& e){snprintf(last_error,sizeof(last_error),"%s",e.what());return 1;}
}
