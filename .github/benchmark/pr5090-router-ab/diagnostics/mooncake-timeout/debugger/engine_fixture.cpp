// CPU-only native library: the debugger must name and unwind this C++ frame.
#include <pthread.h>
namespace mooncake {
class WorkerPool { public: void performPostSend(int); };
void WorkerPool::performPostSend(int) {
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
    pthread_mutex_lock(&mutex);
    for (;;) pthread_cond_wait(&condition, &mutex);
}
}
extern "C" void block_in_engine() {
    mooncake::WorkerPool worker;
    worker.performPostSend(0);
}
