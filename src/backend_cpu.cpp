// Portable multithreaded CPU backend: same contract as the CUDA kernel,
// built from the same include/texture.cuh. Threads pull chunks of the
// linearized volume from an atomic cursor (the CPU analog of the GPU's
// chunk-cursor design) and walk them with the same incremental
// decompose-once-then-carry scheme.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include "kernel.cuh"
#include "texture.cuh"

const char *searchBackendName() { return "CPU"; }

namespace
{

constexpr unsigned long long CHUNK = 1u << 16;

struct Shared
{
    SearchBounds b;
    const BlockInfo *blocks;
    int nBlocks;
    bool modern;
    unsigned long long total;
    long long nx, nz;
    std::atomic<unsigned long long> cursor{0};
    std::atomic<unsigned long long> found{0};
    std::mutex sink;
    std::vector<MatchResult> results;
};

void worker(Shared &S)
{
    std::vector<MatchResult> local;
    unsigned long long myFound = 0;

    for (;;)
    {
        const unsigned long long base = S.cursor.fetch_add(CHUNK);
        if (base >= S.total)
            break;
        const unsigned long long end = std::min(base + CHUNK, S.total);

        unsigned long long r = base;
        int x = S.b.x_min + (int)(r % (unsigned long long)S.nx);
        r /= (unsigned long long)S.nx;
        int z = S.b.z_min + (int)(r % (unsigned long long)S.nz);
        r /= (unsigned long long)S.nz;
        int y = S.b.y_min + (int)r;

        for (unsigned long long g = base; g < end; g++)
        {
            bool ok = true;
            for (int i = 0; i < S.nBlocks; i++)
            {
                const BlockInfo &bi = S.blocks[i];
                const int mod = bi.isSide ? MOD_SIDE : MOD_TOP_BOTTOM;
                const int v = S.modern ? getTextureModern(x + bi.x, y + bi.y, z + bi.z, mod)
                                       : getTextureLegacy(x + bi.x, y + bi.y, z + bi.z, mod);
                if (v != bi.rotation)
                {
                    ok = false;
                    break;
                }
            }
            if (ok)
            {
                myFound++;
                if (local.size() < MAX_MATCHES)
                    local.push_back({x, y, z});
            }

            if (++x > S.b.x_max)
            {
                x = S.b.x_min;
                if (++z > S.b.z_max)
                {
                    z = S.b.z_min;
                    y++;
                }
            }
        }
    }

    S.found.fetch_add(myFound);
    std::lock_guard<std::mutex> lock(S.sink);
    for (const auto &m : local)
    {
        if (S.results.size() >= MAX_MATCHES)
            break;
        S.results.push_back(m);
    }
}

} // namespace

std::vector<MatchResult> runSearch(const SearchBounds &bounds,
                                   const std::vector<BlockInfo> &topsAndBottoms,
                                   const std::vector<BlockInfo> &sides,
                                   int version,
                                   float *kernelMs,
                                   bool *truncated,
                                   unsigned long long *totalMatches)
{
    if (topsAndBottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS)
    {
        std::fprintf(stderr, "Formation exceeds %d blocks per face type\n", MAX_FORMATION_BLOCKS);
        std::exit(1);
    }

    // Tops first: a mod-4 check rejects 75% of candidates, a side check 50%.
    std::vector<BlockInfo> blocks = topsAndBottoms;
    blocks.insert(blocks.end(), sides.begin(), sides.end());
    if (blocks.empty())
    {
        std::fprintf(stderr, "Empty formation\n");
        std::exit(1);
    }

    Shared S;
    S.b = bounds;
    S.blocks = blocks.data();
    S.nBlocks = (int)blocks.size();
    S.modern = version == MODERN_VERSION;
    S.nx = (long long)bounds.x_max - bounds.x_min + 1;
    S.nz = (long long)bounds.z_max - bounds.z_min + 1;
    const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
    const unsigned long long nzU = (unsigned long long)S.nz;
    const unsigned long long nxU = (unsigned long long)S.nx;
    if (ny > (~0ull) / nzU || nxU > (~0ull) / (ny * nzU))
    {
        std::fprintf(stderr, "Search volume too large\n");
        std::exit(1);
    }
    S.total = nxU * ny * nzU;

    const auto t0 = std::chrono::steady_clock::now();

    unsigned n = std::thread::hardware_concurrency();
    if (n == 0)
        n = 4;
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < n; i++)
        pool.emplace_back(worker, std::ref(S));
    for (auto &t : pool)
        t.join();

    const auto t1 = std::chrono::steady_clock::now();
    if (kernelMs)
        *kernelMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const unsigned long long count = S.found.load();
    if (truncated)
        *truncated = count > MAX_MATCHES;
    if (totalMatches)
        *totalMatches = count;

    std::sort(S.results.begin(), S.results.end(), [](const MatchResult &a, const MatchResult &b)
              { return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z); });
    return S.results;
}
