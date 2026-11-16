// Metal compute kernel for the formation search. This file is concatenated
// AFTER src/texture/texture_rules.hpp into one MSL source string at build time (see
// the Makefile's build/tf_shader_src.h rule), so get_texture_modern /
// get_texture_legacy here are the very same code every other backend runs.
// The host side is src/backends/metal/search.mm. device.mm compiles the
// source and injects
//   #define TF_MAX_MATCHES <MAX_MATCHES>
// as a prelude. The bitplane kernels follow in bitplanes.metal.

#include <metal_stdlib>
using namespace metal;

// Selected per-launch like the CUDA template parameter; branches on a
// function constant are resolved at pipeline-creation time.
// 0 modern, 1 legacy, 2 vanilla12, 3 sodium, 4 sodium19 (texture/formation.hpp).
constant int TF_VERSION [[function_constant(0)]];

struct TfParams {
  int x_min, x_max;
  int y_min, y_max;
  int z_max;  // last z of this dispatch
  int z_base; // first z of this dispatch
  int total_blocks;
  int pad;
};

struct TfBlock {
  int x, y, z, w; // w = expected rotation | (compare mask << 8)
};

struct TfMatch {
  int x, y, z;
};

// Counts a match and stores it while there is room. pass_count is the exact
// tally of one pass (the host sums passes into a 64-bit total; a pass covers
// fewer than 2^32 positions). write_idx stops advancing once the buffer is
// full so it can never wrap.
inline void record_match(device TfMatch* out, device atomic_uint* write_idx, device atomic_uint* pass_count,
                         int x, int y, int z) {
  atomic_fetch_add_explicit(pass_count, 1u, memory_order_relaxed);
  if (atomic_load_explicit(write_idx, memory_order_relaxed) < TF_MAX_MATCHES) {
    const uint idx = atomic_fetch_add_explicit(write_idx, 1u, memory_order_relaxed);
    if (idx < TF_MAX_MATCHES) {
      out[idx].x = x;
      out[idx].y = y;
      out[idx].z = z;
    }
  }
}

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
    const int v = texture_from_mix(Version(TF_VERSION), mix, 4);
    if ((v & mask) != expected)
      return;
  }

  record_match(out, write_idx, slice_count, x, y, z);
}
