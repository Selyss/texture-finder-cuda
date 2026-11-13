#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>

#include "cuda_bitplanes.cuh"
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

namespace
{
using namespace cuda_bitplanes;

// Preserve the per-call count, output limit, and ordering for both CUDA paths.
std::vector<MatchResult> collectMatches(const MatchResult *output, const Word *counter,
                                       bool *truncated, Word *totalMatches)
{
    Word count = 0;
    CUDA_CHECK(cudaMemcpy(&count, counter, sizeof(count), cudaMemcpyDeviceToHost));
    if (truncated)
        *truncated = count > MAX_MATCHES;
    if (totalMatches)
        *totalMatches = count;

    std::vector<MatchResult> matches(std::min<Word>(count, MAX_MATCHES));
    if (!matches.empty())
        CUDA_CHECK(cudaMemcpy(matches.data(), output, matches.size() * sizeof(MatchResult),
                              cudaMemcpyDeviceToHost));
    std::sort(matches.begin(), matches.end(), [](const MatchResult &a, const MatchResult &b) {
        return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z);
    });
    return matches;
}

size_t planeBudget()
{
    constexpr size_t defaultBudget = size_t{1} << 30;
    const char *setting = std::getenv("TF_CUDA_BITPLANES_BYTES");
    if (!setting)
        return defaultBudget;
    const std::string_view value(setting);
    size_t bytes = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), bytes);
    return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size()
        ? std::min(bytes, defaultBudget) : 0;
}

bool eligibleForBitplanes(const std::vector<BlockInfo> &blocks)
{
    std::array<int, STRIDE> filterBits{};
    for (const auto &block : blocks)
        filterBits[residue(-static_cast<int64_t>(block.z))] += block.isSide ? 1 : 2;
    // A weak coset can send most of its candidates through scalar verification.
    return *std::min_element(filterBits.begin(), filterBits.end()) >= 4;
}

struct DirectionChecks
{
    std::vector<Check> checks;
    std::vector<VerifyCheck> verify;
    std::array<int, STRIDE + 1> offsets{}, verifyOffsets{};
    PlaneCheck *deviceChecks = nullptr;
    VerifyCheck *deviceVerify = nullptr;
    int *deviceOffsets = nullptr, *deviceVerifyOffsets = nullptr;
    MatchResult *output = nullptr;
    Word *counter = nullptr;
    std::vector<PlaneCheck> slabChecks;
};

std::optional<std::vector<DirectionResult>> tryBitplanes(
    const SearchBounds &bounds, const std::vector<std::vector<BlockInfo>> &formations, float *kernelMs)
{
    int minX = formations.front().front().x, maxX = minX;
    int minZ = formations.front().front().z, maxZ = minZ;
    std::vector<int> yOffsets;
    for (const auto &blocks : formations)
        for (const auto &block : blocks)
        {
            minX = std::min(minX, block.x);
            maxX = std::max(maxX, block.x);
            minZ = std::min(minZ, block.z);
            maxZ = std::max(maxZ, block.z);
            yOffsets.push_back(block.y);
        }

    const int64_t nx = static_cast<int64_t>(bounds.x_max) - bounds.x_min + 1;
    const int64_t ny = static_cast<int64_t>(bounds.y_max) - bounds.y_min + 1;
    const int64_t nz = static_cast<int64_t>(bounds.z_max) - bounds.z_min + 1;
    const int64_t fieldX = static_cast<int64_t>(bounds.x_min) + minX;
    const int64_t width = nx + static_cast<int64_t>(maxX) - minX;
    const int64_t halo = static_cast<int64_t>(maxZ) - minZ;
    std::sort(yOffsets.begin(), yOffsets.end());
    yOffsets.erase(std::unique(yOffsets.begin(), yOffsets.end()), yOffsets.end());
    constexpr int intMin = std::numeric_limits<int>::min();
    constexpr int intMax = std::numeric_limits<int>::max();
    if (nx <= 0 || ny <= 0 || nz <= 0 || width > intMax || ny > intMax || nz > intMax ||
        fieldX < intMin || fieldX + width - 1 > intMax ||
        static_cast<int64_t>(bounds.z_min) + minZ < intMin ||
        static_cast<int64_t>(bounds.z_max) + maxZ > intMax ||
        static_cast<int64_t>(bounds.y_min) + yOffsets.front() < intMin ||
        static_cast<int64_t>(bounds.y_max) + yOffsets.back() > intMax)
        return std::nullopt;

    const size_t fieldWords = ceilDiv(static_cast<int>(width), WORD_BITS) + 1;
    const int candidateWords = ceilDiv(static_cast<int>(nx), WORD_BITS);
    const int segments = ceilDiv(candidateWords, THREADS);
    constexpr size_t resultBytes = MAX_MATCHES * sizeof(MatchResult);
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Leave room for results and the small metadata buffers.
    const size_t reserve = formations.size() * resultBytes + (size_t{16} << 20);
    const size_t budget = std::min(planeBudget(), freeBytes > reserve ? freeBytes - reserve : 0);
    const size_t wordsBudget = budget / sizeof(Planes);
    if (static_cast<size_t>(ny) > wordsBudget / fieldWords)
        return std::nullopt;

    const auto planeY = compactLevels(bounds, yOffsets);
    if (planeY.size() > wordsBudget / fieldWords)
        return std::nullopt;
    const size_t wordsPerZ = planeY.size() * fieldWords;
    const size_t hashedCapacity = wordsBudget / wordsPerZ;
    const int64_t maxRows = static_cast<int64_t>(hashedCapacity) * STRIDE - halo;
    const size_t gridPerZ = static_cast<size_t>(ny) * segments;
    const int64_t plannedRows = std::min<int64_t>({
        nz, DEFAULT_SLAB_ROWS, maxRows, static_cast<int64_t>(intMax / gridPerZ)});
    if (plannedRows <= 0)
        return std::nullopt;
    const int slabRows = static_cast<int>(plannedRows);

    const size_t planeWords = wordsPerZ * ((slabRows + halo + STRIDE - 1) / STRIDE);
    std::vector<DirectionChecks> directions(formations.size());
    for (size_t i = 0; i < formations.size(); ++i)
    {
        auto &direction = directions[i];
        const auto &blocks = formations[i];
        for (int coset = 0; coset < STRIDE; ++coset)
        {
            for (const auto &block : blocks)
            {
                if (residue(-static_cast<int64_t>(block.z)) == coset)
                {
                    const auto level = std::lower_bound(planeY.begin(), planeY.end(), bounds.y_min + block.y);
                    direction.checks.push_back({block.x - minX, static_cast<int>(level - planeY.begin()),
                                      block.z, block.rotation, block.isSide});
                }
                else
                    direction.verify.push_back({static_cast<int64_t>(block.z) * 116129781LL,
                                      static_cast<uint32_t>(block.x) * 3129871u,
                                      block.y, block.rotation, block.isSide ? 1 : 3});
            }
            direction.offsets[coset + 1] = static_cast<int>(direction.checks.size());
            direction.verifyOffsets[coset + 1] = static_cast<int>(direction.verify.size());
        }

    }

    Planes *planes = nullptr;
    const cudaError_t allocation = cudaMalloc(&planes, planeWords * sizeof(Planes));
    if (allocation == cudaErrorMemoryAllocation)
    {
        cudaGetLastError();
        return std::nullopt;
    }
    CUDA_CHECK(allocation);

    auto upload = [](const auto &values) {
        using Element = typename std::decay_t<decltype(values)>::value_type;
        Element *device = nullptr;
        if (!values.empty())
        {
            const size_t bytes = values.size() * sizeof(Element);
            CUDA_CHECK(cudaMalloc(&device, bytes));
            CUDA_CHECK(cudaMemcpy(device, values.data(), bytes, cudaMemcpyHostToDevice));
        }
        return device;
    };
    auto *deviceY = upload(planeY);
    for (auto &direction : directions)
    {
        direction.slabChecks.resize(direction.checks.size());
        CUDA_CHECK(cudaMalloc(&direction.deviceChecks, direction.slabChecks.size() * sizeof(PlaneCheck)));
        direction.deviceVerify = upload(direction.verify);
        direction.deviceOffsets = upload(direction.offsets);
        direction.deviceVerifyOffsets = upload(direction.verifyOffsets);
        CUDA_CHECK(cudaMalloc(&direction.output, resultBytes));
        CUDA_CHECK(cudaMalloc(&direction.counter, sizeof(Word)));
        CUDA_CHECK(cudaMemset(direction.counter, 0, sizeof(Word)));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int64_t offset = 0; offset < nz;)
    {
        const int rows = static_cast<int>(std::min<int64_t>(slabRows, nz - offset));
        SearchBounds slab = bounds;
        slab.z_min = static_cast<int>(bounds.z_min + offset);
        slab.z_max = slab.z_min + (rows - 1);
        const int64_t haloStart = static_cast<int64_t>(slab.z_min) + minZ;
        const int64_t haloEnd = static_cast<int64_t>(slab.z_max) + maxZ;
        const int64_t firstZ = haloStart + residue(-haloStart);
        const int hashedRows = firstZ <= haloEnd ? static_cast<int>((haloEnd - firstZ) / STRIDE + 1) : 0;
        const Layout layout{slab, static_cast<int>(nx), static_cast<int>(ny), rows,
                            static_cast<int>(fieldX), static_cast<int>(width), static_cast<int>(fieldWords),
                            candidateWords, static_cast<int>(planeY.size()), hashedRows, firstZ};
        const size_t words = wordsPerZ * hashedRows;
        if (words)
        {
            fillPlanes<true><<<(words + THREADS - 1) / THREADS, THREADS>>>(layout, deviceY, words, planes);
            CUDA_CHECK(cudaGetLastError());
        }
        for (auto &direction : directions)
        {
            preparePlaneChecks(layout, direction.checks, direction.offsets.data(), direction.slabChecks);
            // Copies and kernels are ordered on the default stream.
            CUDA_CHECK(cudaMemcpy(direction.deviceChecks, direction.slabChecks.data(),
                                  direction.slabChecks.size() * sizeof(PlaneCheck), cudaMemcpyHostToDevice));
            matchWords<MatchMode::Search, true, true><<<gridPerZ * rows, THREADS>>>(
                layout, direction.deviceChecks, direction.deviceOffsets, planes, nullptr, nullptr, 0,
                direction.deviceVerify, direction.deviceVerifyOffsets,
                direction.output, direction.counter, MAX_MATCHES);
            CUDA_CHECK(cudaGetLastError());
        }
        offset += rows;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (kernelMs)
        CUDA_CHECK(cudaEventElapsedTime(kernelMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<DirectionResult> results(directions.size());
    for (size_t i = 0; i < directions.size(); ++i)
    {
        const auto &direction = directions[i];
        auto &result = results[i];
        result.matches = collectMatches(direction.output, direction.counter, &result.truncated, &result.total);
        CUDA_CHECK(cudaFree(direction.counter));
        CUDA_CHECK(cudaFree(direction.output));
        CUDA_CHECK(cudaFree(direction.deviceVerifyOffsets));
        CUDA_CHECK(cudaFree(direction.deviceOffsets));
        CUDA_CHECK(cudaFree(direction.deviceVerify));
        CUDA_CHECK(cudaFree(direction.deviceChecks));
    }
    CUDA_CHECK(cudaFree(deviceY));
    CUDA_CHECK(cudaFree(planes));
    return results;
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

    if (const char *enabled = std::getenv("TF_CUDA_BITPLANES");
        enabled && std::string_view(enabled) == "1")
    {
        const bool trace = std::getenv("TF_CUDA_BITPLANES_TRACE") != nullptr;
        if (version == MODERN_VERSION)
        {
            std::vector<std::vector<BlockInfo>> formations{topsAndBottoms};
            formations.front().insert(formations.front().end(), sides.begin(), sides.end());
            if (eligibleForBitplanes(formations.front()))
            {
                if (auto results = tryBitplanes(bounds, formations, kernelMs))
                {
                    if (trace)
                        std::fputs("CUDA search: bitplanes\n", stderr);
                    auto &result = results->front();
                    if (truncated)
                        *truncated = result.truncated;
                    if (totalMatches)
                        *totalMatches = result.total;
                    return std::move(result.matches);
                }
            }
        }
        if (trace)
            std::fputs("CUDA search: baseline fallback\n", stderr);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_blocks, blocks.data(), blocks.size() * sizeof(DevBlock)));

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

    auto matches = collectMatches(d_out, d_count, truncated, totalMatches);
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_cursor));
    return matches;
}

bool sharedSearchEnabled(int version)
{
    const char *enabled = std::getenv("TF_CUDA_BITPLANES");
    return version == MODERN_VERSION && enabled && std::string_view(enabled) == "1";
}

std::vector<DirectionResult> runSharedSearch(const SearchBounds &bounds,
                                             const std::vector<SearchFormation> &formations,
                                             int version, double &kernelSec)
{
    std::vector<DirectionResult> results(formations.size());
    std::vector<size_t> sharedIndices;
    std::vector<std::vector<BlockInfo>> eligible;
    if (sharedSearchEnabled(version))
    {
        for (size_t i = 0; i < formations.size(); ++i)
        {
            const auto &formation = formations[i];
            if (formation.tops.size() > MAX_FORMATION_BLOCKS || formation.sides.size() > MAX_FORMATION_BLOCKS)
                continue; // runSearch reports the input error.
            auto blocks = formation.tops;
            blocks.insert(blocks.end(), formation.sides.begin(), formation.sides.end());
            if (eligibleForBitplanes(blocks))
            {
                sharedIndices.push_back(i);
                eligible.push_back(std::move(blocks));
            }
        }
    }
    std::optional<std::vector<DirectionResult>> shared;
    if (eligible.size() > 1)
    {
        float ms = 0;
        shared = tryBitplanes(bounds, eligible, &ms);
        if (shared)
        {
            kernelSec += ms / 1000.0;
            if (std::getenv("TF_CUDA_BITPLANES_TRACE"))
                std::fprintf(stderr, "CUDA search: bitplanes shared (%zu directions)\n", shared->size());
        }
    }
    size_t nextShared = 0;
    for (size_t i = 0; i < formations.size(); ++i)
    {
        if (shared && nextShared < sharedIndices.size() && sharedIndices[nextShared] == i)
        {
            results[i] = std::move((*shared)[nextShared++]);
            continue;
        }
        auto &result = results[i];
        float ms = 0;
        result.matches = runSearch(bounds, formations[i].tops, formations[i].sides,
                                   version, &ms, &result.truncated, &result.total);
        kernelSec += ms / 1000.0;
    }
    return results;
}
