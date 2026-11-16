#pragma once

#include "bitplanes/match.hpp"

// The kernels of the bitplane search, for the modern version. The variants
// that were measured against them live in
// experiments/rng-reduction/diagnostic_kernels.cuh. The kernels are static
// because the search and the harness each compile this header.
namespace cuda_bitplanes {

using namespace bitplanes;

// Hashes one field word: 64 x positions on one hashed z row of one y level.
static __global__ void fill_planes(Layout layout, const int* plane_y, size_t count, Planes* planes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const int word = index % layout.field_words;
  const size_t row = index / layout.field_words;
  const int z_index = row % layout.hashed_rows;
  planes[index] = field_word(Version::modern, layout, plane_y[row / layout.hashed_rows], z_index, word);
}

// Each surviving bit belongs to exactly one thread, which hashes the blocks
// outside the candidate's coset and records the full matches.
__device__ __forceinline__ void verify_survivors(const Layout& layout, Word survivors, size_t y, size_t z_row,
                                                 int word, const VerifyCheck* verify,
                                                 const int* verify_offsets, MatchResult* matches,
                                                 Word* total_matches, Word capacity) {
  const int candidate_y = layout.bounds.y_min + static_cast<int>(y);
  const int z = layout.bounds.z_min + static_cast<int>(z_row);
  const int coset = residue(z);
  const int64_t z_term = static_cast<int64_t>(z) * 116129781LL;
  for (; survivors; survivors &= survivors - 1) {
    const int bit = __ffsll(survivors) - 1;
    const int x = layout.bounds.x_min + word * WORD_BITS + bit;
    if (survivor_matches(Version::modern, verify, verify_offsets[coset], verify_offsets[coset + 1],
                         static_cast<uint32_t>(x) * 3129871u, z_term, candidate_y)) {
      const Word index = atomicAdd(total_matches, 1ull);
      if (index < capacity)
        matches[index] = {x, candidate_y, z};
    }
  }
}

// One thread per candidate word, one block per THREADS words of a (y, z) row.
static __global__ void match_words(Layout layout, const PlaneCheck* checks, const int* coset_offsets,
                                   const Planes* planes, const VerifyCheck* verify, const int* verify_offsets,
                                   MatchResult* matches, Word* total_matches, Word capacity) {
  const int segments = ceil_div(layout.candidate_words, THREADS);
  const size_t row = blockIdx.x / segments;
  const int word = (blockIdx.x % segments) * THREADS + threadIdx.x;
  if (word >= layout.candidate_words)
    return;
  const size_t y = row / layout.nz;
  const size_t z_row = row % layout.nz;
  const Word mask = filter_word(layout, checks, coset_offsets, planes, y, z_row, word);
  verify_survivors(layout, mask, y, z_row, word, verify, verify_offsets, matches, total_matches, capacity);
}

} // namespace cuda_bitplanes
