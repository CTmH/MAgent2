// Build from the repository root (add -fopenmp to exercise OpenMP):
// g++ -std=c++11 -I . tests/test_reset_buffers.cc src/gridworld/AgentType.cc \
//     src/gridworld/GridWorld.cc src/gridworld/Map.cc src/gridworld/RenderGenerator.cc \
//     src/gridworld/RewardEngine.cc src/utility/utility.cc -o /tmp/check-reset
#include <cassert>
#include <cstdlib>
#include <new>
#include "src/gridworld/GridWorld.h"
static long arrays = 0;
void* operator new[](std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    ++arrays;
    return p;
}
void operator delete[](void* p) noexcept { if (p) { --arrays; std::free(p); } }
void operator delete[](void* p, std::size_t) noexcept { operator delete[](p); }
int main() {
    long baseline = arrays;
    {
        magent::gridworld::GridWorld env;
        int size = 100;
        env.set_config("map_width", &size);
        env.set_config("map_height", &size);
        env.reset();
        long initial = arrays;
        for (int i=0; i<100; ++i) { env.reset(); assert(arrays == initial); }
        size = 12;
        env.set_config("map_width", &size);
        env.set_config("map_height", &size);
        env.reset();
        assert(arrays == initial - 2);
        size = 100;
        env.set_config("map_width", &size);
        env.set_config("map_height", &size);
        env.reset();
        assert(arrays == initial);
    }
    assert(arrays == baseline);
}
