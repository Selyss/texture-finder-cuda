#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "kernel.cuh"
#include "texture.cuh"

#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess)                                                  \
        {                                                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                               \
            std::exit(2);                                                         \
        }                                                                         \
    } while (0)

const char *searchBackendName() { return "CUDA"; }

// Tuning knobs, overridable at compile time (test/bench.sh sweeps these;
// defaults are the winners on an RTX 3090).
#ifndef TF_BLOCK_THREADS
#define TF_BLOCK_THREADS 256
#endif
// Positions each lane pulls per work chunk; one chunk = 32x this.
#ifndef TF_CHUNK_PER_LANE
#define TF_CHUNK_PER_LANE 1024
#endif

constexpr int WARP = 32;

static_assert(TF_CHUNK_PER_LANE > 0, "zero chunk size would livelock the chunk cursor");
static_assert(TF_BLOCK_THREADS % WARP == 0, "partial warps break the full-mask warp intrinsics");

// One formation block with its hash terms precomputed: the position hash's
// x and z multiplies distribute over the block offset under wrapping (see
// coordRandomFromParts), so per evaluation the kernel only adds these to the
// candidate's own terms instead of multiplying.
// w = expected rotation | (compare mask << 8); mask is 3 for top faces and 1
// for side faces, because for mods 2 and 4 both generators reduce to the
// mod-4 value masked down (see texture.cuh: side value == top roll % 2).
struct DevBlock
{
    long long zTerm; // bz * 116129781
    unsigned int xTerm; // bx * 3129871 (mod 2^32)
    int dy;
    int w;
};

__constant__ DevBlock d_blocks[2 * MAX_FORMATION_BLOCKS];

// A maximum-size formation must fit the default dynamic shared memory limit.
static_assert(2 * MAX_FORMATION_BLOCKS * sizeof(DevBlock) <= 48 * 1024,
              "formation no longer fits default dynamic shared memory");

// Warp-compacted search. The naive one-thread-per-position kernel wastes ~2/3
// of the issue slots: with a 1-in-4 pass rate per check, a warp's slowest lane
// dictates ~4.3 evaluations per batch while the average position needs ~1.33.
// Here every lane keeps its own candidate position and block cursor; the
// moment a candidate fails, the lane pulls a fresh position from its stripe
// of the current warp chunk. Lanes therefore stay busy evaluating block 0/1
// almost all the time. Warps grab chunks from a global cursor; within a chunk
// lane L owns linear positions chunkBase + pull*32 + L, so every position in
// [0, total) is evaluated exactly once, no matter how the grid is sized.
template <int VERSION>
__global__ void __launch_bounds__(TF_BLOCK_THREADS)
    matchFormationKernel(SearchBounds b, int totalBlocks, unsigned long long total,
                         unsigned long long *chunkCursor,
                         MatchResult *out, unsigned long long *outCount)
{
    // Formation goes to shared memory: compacted lanes sit at different block
    // indices, and divergent-address reads serialize in constant memory but
    // not in shared memory. Dynamically sized so small formations cost no
    // occupancy.
    extern __shared__ DevBlock sBlocks[];
    for (int i = (int)threadIdx.x; i < totalBlocks; i += (int)blockDim.x)
        sBlocks[i] = d_blocks[i];
    __syncthreads();

    const int lane = threadIdx.x & (WARP - 1);
    const long long nx = (long long)b.x_max - b.x_min + 1;
    const long long nz = (long long)b.z_max - b.z_min + 1;
    const unsigned long long chunkSize = (unsigned long long)WARP * TF_CHUNK_PER_LANE;

    while (true)
    {
        // The warp takes a chunk; lane L owns linear positions
        // base + pull*32 + L, so every index in [0, total) is evaluated
        // exactly once across all warps.
        unsigned long long base = 0;
        if (lane == 0)
            base = atomicAdd(chunkCursor, chunkSize);
        base = __shfl_sync(0xffffffffu, base, 0);
        if (base >= total)
            return; // uniform exit

        // Number of in-range pulls in this lane's stripe, computed once so
        // the per-pull path needs no 64-bit index math or range compare:
        // positions gLane, gLane+32, ..., gLane+(budget-1)*32 are < total.
        const unsigned long long gLane = base + lane;
        int budget = 0;
        if (gLane < total)
        {
            const unsigned long long remaining = total - gLane;
            const unsigned long long pulls = (remaining + WARP - 1) / WARP;
            budget = (int)(pulls < TF_CHUNK_PER_LANE ? pulls : TF_CHUNK_PER_LANE);
        }

        int x = 0, y = 0, z = 0;     // candidate origin (valid while bi >= 0)
        unsigned int hx = 0;         // x * 3129871 (mod 2^32)
        long long hz = 0;            // z * 116129781
        int bi = -1;                 // next block to check; -1 = no live candidate
        bool first = true;

        // Each lane processes its stripe independently: the moment a
        // candidate fails, the lane pulls a fresh one, so lanes stay busy on
        // the highly selective first blocks instead of idling while a
        // warp-mate finishes a deep near-match.
        while (bi >= 0 || budget > 0)
        {
            if (bi < 0)
            {
                budget--;
                if (first)
                {
                    // First pull of a chunk: full decompose (amortized away).
                    first = false;
                    unsigned long long r = gLane;
                    x = b.x_min + (int)(r % (unsigned long long)nx);
                    r /= (unsigned long long)nx;
                    z = b.z_min + (int)(r % (unsigned long long)nz);
                    r /= (unsigned long long)nz;
                    y = b.y_min + (int)r;
                    hx = (unsigned int)x * 3129871u;
                    hz = (long long)z * 116129781LL;
                }
                else
                {
                    // Consecutive pulls advance exactly WARP linear steps;
                    // the x hash term advances by a compile-time constant
                    // (wrapping add), so the common path multiplies nothing.
                    x += WARP;
                    hx += (unsigned int)WARP * 3129871u;
                    if (x > b.x_max)
                    {
                        do
                        {
                            x -= (int)nx;
                            z++;
                            if (z > b.z_max)
                            {
                                z -= (int)nz;
                                y++;
                            }
                        } while (x > b.x_max);
                        hx = (unsigned int)x * 3129871u;
                        hz = (long long)z * 116129781LL;
                    }
                }
                bi = 0;
            }

            const DevBlock blk = sBlocks[bi];
            const int expected = blk.w & 0xff;
            const int mask = blk.w >> 8;
            const int64_t mix = coordMixFromParts(hx + blk.xTerm, hz + blk.zTerm, y + blk.dy);
            int v;
            if (VERSION == LEGACY_VERSION)
                v = legacyFromCoordRandom(mix >> 16, 4);
            else if (VERSION == VANILLA12_VERSION)
                v = vanilla12FromMix(mix, 4);
            else if (VERSION == SODIUM_VERSION)
                v = sodiumFromCoordRandom(mix >> 16, 4);
            else if (VERSION == SODIUM19_VERSION)
                v = sodium19FromCoordRandom(mix >> 16, 4);
            else
                v = modernFromCoordRandom(mix >> 16, 4);
            if ((v & mask) != expected)
            {
                bi = -1;
            }
            else if (++bi == totalBlocks)
            {
                // 64-bit: a dense weak-formation search can exceed 2^32
                // matches, which would wrap a 32-bit counter and restart
                // idx at 0, overwriting the buffer.
                const unsigned long long idx = atomicAdd(outCount, 1ull);
                if (idx < MAX_MATCHES)
                    out[idx] = {x, y, z};
                bi = -1;
            }
        }
    }
}

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

    // Tops first: a mod-4 check rejects 75% of candidates, a side check only
    // 50%, so the most selective checks run earliest.
    std::vector<DevBlock> blocks;
    blocks.reserve(topsAndBottoms.size() + sides.size());
    for (const auto &t : topsAndBottoms)
        blocks.push_back({(long long)t.z * 116129781LL, (unsigned int)t.x * 3129871u,
                          t.y, t.rotation | (3 << 8)});
    for (const auto &s : sides)
        blocks.push_back({(long long)s.z * 116129781LL, (unsigned int)s.x * 3129871u,
                          s.y, s.rotation | (1 << 8)});
    if (blocks.empty())
    {
        std::fprintf(stderr, "Empty formation\n");
        std::exit(1);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_blocks, blocks.data(), blocks.size() * sizeof(DevBlock)));

    const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
    const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
    const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
    // Two-step overflow guard: ny*nz itself must not overflow before it is
    // used as a divisor (nx/ny/nz are each >= 1 by construction).
    if (ny > (~0ull) / nz || nx > (~0ull) / (ny * nz))
    {
        std::fprintf(stderr, "Search volume too large\n");
        std::exit(1);
    }
    const unsigned long long total = nx * ny * nz;

    MatchResult *d_out = nullptr;
    unsigned long long *d_count = nullptr;
    unsigned long long *d_cursor = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, MAX_MATCHES * sizeof(MatchResult)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_cursor, 0, sizeof(unsigned long long)));

    // Persistent-thread launch sized to fill the device exactly once.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const size_t sharedBytes = blocks.size() * sizeof(DevBlock);

    // Compile-time version dispatch: resolve the kernel instantiation once.
    using KernelFn = void (*)(SearchBounds, int, unsigned long long,
                              unsigned long long *, MatchResult *, unsigned long long *);
    KernelFn kernel;
    switch (version)
    {
    case LEGACY_VERSION:
        kernel = matchFormationKernel<LEGACY_VERSION>;
        break;
    case VANILLA12_VERSION:
        kernel = matchFormationKernel<VANILLA12_VERSION>;
        break;
    case SODIUM_VERSION:
        kernel = matchFormationKernel<SODIUM_VERSION>;
        break;
    case SODIUM19_VERSION:
        kernel = matchFormationKernel<SODIUM19_VERSION>;
        break;
    default:
        kernel = matchFormationKernel<MODERN_VERSION>;
        break;
    }

    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, kernel, TF_BLOCK_THREADS, sharedBytes));
    const unsigned long long chunkSize = (unsigned long long)WARP * TF_CHUNK_PER_LANE;
    const unsigned long long chunksNeeded = total / chunkSize + (total % chunkSize != 0);
    const unsigned long long warpsNeeded = chunksNeeded; // one chunk in flight per warp
    unsigned long long grid = (unsigned long long)props.multiProcessorCount * (blocksPerSM > 0 ? blocksPerSM : 1);
    const unsigned long long gridForWork = (warpsNeeded * WARP + TF_BLOCK_THREADS - 1) / TF_BLOCK_THREADS;
    if (grid > gridForWork)
        grid = gridForWork; // tiny searches don't need the whole device
    if (grid == 0)
        grid = 1;

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    kernel<<<(unsigned int)grid, TF_BLOCK_THREADS, sharedBytes>>>(
        bounds, (int)blocks.size(), total, d_cursor, d_out, d_count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    if (kernelMs)
    {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        *kernelMs = ms;
    }
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    unsigned long long count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    const unsigned int stored = (unsigned int)std::min(count, (unsigned long long)MAX_MATCHES);
    if (truncated)
        *truncated = count > MAX_MATCHES;
    if (totalMatches)
        *totalMatches = count;

    std::vector<MatchResult> matches(stored);
    if (stored > 0)
        CUDA_CHECK(cudaMemcpy(matches.data(), d_out, stored * sizeof(MatchResult),
                              cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_cursor));

    std::sort(matches.begin(), matches.end(), [](const MatchResult &a, const MatchResult &b)
              { return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z); });
    return matches;
}
