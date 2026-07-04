// Metal compute kernel for the formation search. This file is concatenated
// AFTER include/texture.cuh into one MSL source string at build time (see
// the Makefile's build/tf_shader_src.h rule), so getTextureModern /
// getTextureLegacy here are the very same code every other backend runs.
// The host side is src/backend_metal.mm, which also injects
//   #define TF_MAX_MATCHES <MAX_MATCHES>
// as a prelude and asserts it matches kernel.cuh.

#include <metal_stdlib>
using namespace metal;

// Selected per-launch like the CUDA template parameter; branches on a
// function constant are resolved at pipeline-creation time.
// 0 modern, 1 legacy, 2 vanilla12, 3 sodium, 4 sodium19 (blockinfo.cuh).
constant int TF_VERSION [[function_constant(0)]];

struct TfParams
{
    int x_min, x_max;
    int y_min, y_max;
    int z_max;
    int zBase;       // z origin of this slice
    int totalBlocks;
    int pad;
};

struct TfBlock
{
    int x, y, z, w; // w = expected rotation | (compare mask << 8)
};

struct TfMatch
{
    int x, y, z;
};

kernel void matchFormation(device const TfBlock *blocks [[buffer(0)]],
                           constant TfParams &P [[buffer(1)]],
                           device TfMatch *out [[buffer(2)]],
                           device atomic_uint *writeIdx [[buffer(3)]],
                           device atomic_uint *sliceCount [[buffer(4)]],
                           uint3 t [[thread_position_in_grid]])
{
    const int x = P.x_min + (int)t.x;
    const int y = P.y_min + (int)t.y;
    const int z = P.zBase + (int)t.z;
    // Dispatched in whole threadgroups; the ragged edge is cut here.
    if (x > P.x_max || y > P.y_max || z > P.z_max)
        return;

    for (int i = 0; i < P.totalBlocks; i++)
    {
        const TfBlock b = blocks[i];
        const int expected = b.w & 0xff;
        const int mask = b.w >> 8;
        const int64_t mix = coordMix(x + b.x, y + b.y, z + b.z);
        int v;
        if (TF_VERSION == 1)
            v = legacyFromCoordRandom(mix >> 16, 4);
        else if (TF_VERSION == 2)
            v = vanilla12FromMix(mix, 4);
        else if (TF_VERSION == 3)
            v = sodiumFromCoordRandom(mix >> 16, 4);
        else if (TF_VERSION == 4)
            v = sodium19FromCoordRandom(mix >> 16, 4);
        else
            v = modernFromCoordRandom(mix >> 16, 4);
        if ((v & mask) != expected)
            return;
    }

    // sliceCount is the exact per-slice tally (host sums slices into a
    // 64-bit total; slices are capped well under 2^32 positions). writeIdx
    // stops advancing once the buffer is full so it can never wrap.
    atomic_fetch_add_explicit(sliceCount, 1u, memory_order_relaxed);
    if (atomic_load_explicit(writeIdx, memory_order_relaxed) < TF_MAX_MATCHES)
    {
        const uint idx = atomic_fetch_add_explicit(writeIdx, 1u, memory_order_relaxed);
        if (idx < TF_MAX_MATCHES)
        {
            out[idx].x = x;
            out[idx].y = y;
            out[idx].z = z;
        }
    }
}
