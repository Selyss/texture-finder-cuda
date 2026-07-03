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

__constant__ BlockInfo d_topsAndBottoms[MAX_FORMATION_BLOCKS];
__constant__ BlockInfo d_sides[MAX_FORMATION_BLOCKS];

// If a search legitimately produces more matches than this, the formation is
// far too small to identify a location anyway; the host warns and asks for a
// tighter search instead of allocating gigabytes.
constexpr unsigned int MAX_MATCHES = 1u << 20;

template <bool MODERN>
__device__ __forceinline__ int textureAt(int x, int y, int z, int mod)
{
    return MODERN ? getTextureModern(x, y, z, mod) : getTextureLegacy(x, y, z, mod);
}

// Grid-stride in all three dimensions: every coordinate in the inclusive
// bounds is visited no matter how the grid is sized, so no launch-geometry
// choice can silently skip coordinates (the old fixed-grid launch missed the
// last x/z when the range width was a multiple of the block size, and would
// have exceeded the 65535 grid-dimension limit on large z ranges).
template <bool MODERN>
__global__ void matchFormationKernel(SearchBounds b, int tbSize, int sideSize,
                                     MatchResult *out, unsigned int *outCount)
{
    const int strideX = (int)(blockDim.x * gridDim.x);
    const int strideY = (int)(blockDim.y * gridDim.y);
    const int strideZ = (int)(blockDim.z * gridDim.z);

    for (int y = b.y_min + (int)(blockIdx.y * blockDim.y + threadIdx.y); y <= b.y_max; y += strideY)
        for (int z = b.z_min + (int)(blockIdx.z * blockDim.z + threadIdx.z); z <= b.z_max; z += strideZ)
            for (int x = b.x_min + (int)(blockIdx.x * blockDim.x + threadIdx.x); x <= b.x_max; x += strideX)
            {
                bool ok = true;

                for (int i = 0; i < tbSize; i++)
                {
                    const BlockInfo &bi = d_topsAndBottoms[i];
                    if (textureAt<MODERN>(x + bi.x, y + bi.y, z + bi.z, MOD_TOP_BOTTOM) != bi.rotation)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    continue;

                for (int i = 0; i < sideSize; i++)
                {
                    const BlockInfo &bi = d_sides[i];
                    if (textureAt<MODERN>(x + bi.x, y + bi.y, z + bi.z, MOD_SIDE) != bi.rotation)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    continue;

                unsigned int idx = atomicAdd(outCount, 1u);
                if (idx < MAX_MATCHES)
                    out[idx] = {x, y, z};
            }
}

std::vector<MatchResult> runSearch(const SearchBounds &bounds,
                                   const std::vector<BlockInfo> &topsAndBottoms,
                                   const std::vector<BlockInfo> &sides,
                                   int version,
                                   float *kernelMs,
                                   bool *truncated)
{
    if (topsAndBottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS)
    {
        std::fprintf(stderr, "Formation exceeds %d blocks per face type\n", MAX_FORMATION_BLOCKS);
        std::exit(1);
    }

    if (!topsAndBottoms.empty())
        CUDA_CHECK(cudaMemcpyToSymbol(d_topsAndBottoms, topsAndBottoms.data(),
                                      topsAndBottoms.size() * sizeof(BlockInfo)));
    if (!sides.empty())
        CUDA_CHECK(cudaMemcpyToSymbol(d_sides, sides.data(), sides.size() * sizeof(BlockInfo)));

    MatchResult *d_out = nullptr;
    unsigned int *d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, MAX_MATCHES * sizeof(MatchResult)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    const long long nx = (long long)bounds.x_max - bounds.x_min + 1;
    const long long ny = (long long)bounds.y_max - bounds.y_min + 1;
    const long long nz = (long long)bounds.z_max - bounds.z_min + 1;

    auto clampLL = [](long long v, long long lo, long long hi)
    { return (unsigned int)std::max(lo, std::min(v, hi)); };

    const dim3 threadsPerBlock(256, 1, 1);
    const dim3 numBlocks(clampLL((nx + 255) / 256, 1, 4096),
                         clampLL(ny, 1, 64),
                         clampLL(nz, 1, 64));

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    if (version == MODERN_VERSION)
        matchFormationKernel<true><<<numBlocks, threadsPerBlock>>>(
            bounds, (int)topsAndBottoms.size(), (int)sides.size(), d_out, d_count);
    else
        matchFormationKernel<false><<<numBlocks, threadsPerBlock>>>(
            bounds, (int)topsAndBottoms.size(), (int)sides.size(), d_out, d_count);

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

    unsigned int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    const unsigned int stored = std::min(count, MAX_MATCHES);
    if (truncated)
        *truncated = count > MAX_MATCHES;

    std::vector<MatchResult> matches(stored);
    if (stored > 0)
        CUDA_CHECK(cudaMemcpy(matches.data(), d_out, stored * sizeof(MatchResult),
                              cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_count));

    std::sort(matches.begin(), matches.end(), [](const MatchResult &a, const MatchResult &b)
              { return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z); });
    return matches;
}
