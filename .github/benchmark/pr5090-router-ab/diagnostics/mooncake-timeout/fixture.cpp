#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>

static std::atomic<int> original_calls{0};
static std::atomic<int> callback_calls{0};
namespace mooncake {
struct Status {
    uint16_t code;
    void* message;
    ~Status() {}  // Nontrivial 16-byte return exercises the actual sret ABI.
};
class MultiTransport {
 public:
    Status freeBatchID(uint64_t id, const std::function<void()>& callback);
};
Status MultiTransport::freeBatchID(uint64_t id, const std::function<void()>& callback) {
    ++original_calls;
    callback();
    return Status{7, reinterpret_cast<void*>(id)};
}
}
template<class T, size_t N> void put(std::array<unsigned char,N>& a,size_t i,T v) {
    std::memcpy(a.data()+i,&v,sizeof(v));
}
extern "C" int run_case(int kind) {
    std::array<unsigned char,96> batch{};
    std::array<unsigned char,152*160> tasks{};
    const size_t task_count = kind == 3 ? 160 : 2;
    const size_t task_bytes = 152 * task_count;
    std::array<unsigned char,72> request{};
    std::array<unsigned char,120> slice{};
    uint64_t slice_ptr = reinterpret_cast<uint64_t>(slice.data());
    uint64_t id = reinterpret_cast<uint64_t>(batch.data());
    put(batch,0,id); put(batch,8,uint64_t{task_count});
    put(batch,16,reinterpret_cast<uint64_t>(tasks.data()));
    put(batch,24,reinterpret_cast<uint64_t>(tasks.data()+task_bytes));
    put(batch,32,reinterpret_cast<uint64_t>(tasks.data()+task_bytes));
    batch[0x50]=1;
    if(kind==0) { batch[0x3a]=1; put(batch,0x48,uint64_t{2}); }
    for(size_t i=0;i<task_count;i++) {
        const bool pending = (kind==1 && i==1) || (kind==3 && i>=140);
        put(tasks,i*152,uint64_t{4});
        put(tasks,i*152+8,uint64_t{pending?3u:4u});
        tasks[i*152+24]=pending?0:1;
        put(tasks,i*152+32,(uint64_t{1}<<63)+(pending?3:4));
        put(tasks,i*152+40,uint64_t{1024});
        put(tasks,i*152+56,uint64_t{1024});
        put(tasks,i*152+64,id);
        put(tasks,i*152+112,reinterpret_cast<uint64_t>(request.data()));
        put(tasks,i*152+120,uint64_t{1});
    }
    put(tasks,152+128,reinterpret_cast<uint64_t>(&slice_ptr));
    put(tasks,152+136,reinterpret_cast<uint64_t>(&slice_ptr)+8);
    put(slice,0,uint64_t{0x100000});put(slice,8,uint64_t{256});
    put(slice,104,int{1});put(slice,112,reinterpret_cast<uint64_t>(tasks.data()+152));
    put(request,0,int{1});put(request,8,uint64_t{0x100000});
    put(request,16,uint64_t{123});put(request,24,uint64_t{0x200000});put(request,32,uint64_t{1024});
    auto before_batch=batch;auto before_tasks=tasks;
    mooncake::MultiTransport transport;
    auto returned=transport.freeBatchID(id,[]{++callback_calls;});
    return returned.code==7 && returned.message==reinterpret_cast<void*>(id) &&
        before_batch==batch && before_tasks==tasks ? 0 : 1;
}
extern "C" int calls() { return original_calls.load(); }
extern "C" int callbacks() { return callback_calls.load(); }
