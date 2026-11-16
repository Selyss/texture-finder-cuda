// Metal compute kernel for the formation search. This file is concatenated
// AFTER src/texture/texture_rules.hpp into one MSL source string at build time (see
// the Makefile's build/tf_shader_src.h rule), so get_texture_modern /
// get_texture_legacy here are the very same code every other backend runs.
// The host side is src/backends/metal/search.mm, which also injects
//   #define TF_MAX_MATCHES <MAX_MATCHES>
// as a prelude and asserts it matches search/backend.hpp.

#include <metal_stdlib>
using namespace metal;

// Selected per-launch like the CUDA template parameter; branches on a
// function constant are resolved at pipeline-creation time.
// 0 modern, 1 legacy, 2 vanilla12, 3 sodium, 4 sodium19 (texture/formation.hpp).
constant int TF_VERSION [[function_constant(0)]];

struct TfParams {
  int x_min, x_max;
  int y_min, y_max;
  int z_max;
  int z_base; // z origin of this slice
  int total_blocks;
  int pad;
};

struct TfBlock {
  int x, y, z, w; // w = expected rotation | (compare mask << 8)
};

struct TfMatch {
  int x, y, z;
};

kernel void match_formation(device const TfBlock* blocks [[buffer(0)]], constant TfParams& P [[buffer(1)]],
                            device TfMatch* out [[buffer(2)]], device atomic_uint* write_idx [[buffer(3)]],
                            device atomic_uint* slice_count [[buffer(4)]],
                            uint3 t [[thread_position_in_grid]]) {
  const int x = P.x_min + (int)t.x;
  const int y = P.y_min + (int)t.y;
  const int z = P.z_base + (int)t.z;
  // Dispatched in whole threadgroups; the ragged edge is cut here.
  if (x > P.x_max || y > P.y_max || z > P.z_max)
    return;

  for (int i = 0; i < P.total_blocks; i++) {
    const TfBlock b = blocks[i];
    const int expected = b.w & 0xff;
    const int mask = b.w >> 8;
    const int64_t mix = coord_mix(x + b.x, y + b.y, z + b.z);
    int v;
    if (TF_VERSION == 1)
      v = legacy_from_coord_random(mix >> 16, 4);
    else if (TF_VERSION == 2)
      v = vanilla12_from_mix(mix, 4);
    else if (TF_VERSION == 3)
      v = sodium_from_coord_random(mix >> 16, 4);
    else if (TF_VERSION == 4)
      v = sodium19_from_coord_random(mix >> 16, 4);
    else
      v = modern_from_coord_random(mix >> 16, 4);
    if ((v & mask) != expected)
      return;
  }

  // slice_count is the exact per-slice tally (host sums slices into a
  // 64-bit total; slices are capped well under 2^32 positions). write_idx
  // stops advancing once the buffer is full so it can never wrap.
  atomic_fetch_add_explicit(slice_count, 1u, memory_order_relaxed);
  if (atomic_load_explicit(write_idx, memory_order_relaxed) < TF_MAX_MATCHES) {
    const uint idx = atomic_fetch_add_explicit(write_idx, 1u, memory_order_relaxed);
    if (idx < TF_MAX_MATCHES) {
      out[idx].x = x;
      out[idx].y = y;
      out[idx].z = z;
    }
  }
}
