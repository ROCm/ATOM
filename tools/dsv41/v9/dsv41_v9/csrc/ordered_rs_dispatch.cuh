void dispatchDsv41RSOrdered(hipStream_t stream, opus::bf16_t* input,
                          opus::bf16_t* output, const uint8_t* order_codes)
{
    if(world_size_!=4) throw std::runtime_error("DSV41 ordered RS requires TP4");
    RankData* ptrs=get_buffer_RD(stream,input);
    constexpr int range=(1536*5120)/(4*8);
    dsv41_rs_bf16_ordered<opus::bf16_t><<<80,512,0,stream>>>(
        ptrs,sg_,self_sg_,output,rank_,range,order_codes);
}
