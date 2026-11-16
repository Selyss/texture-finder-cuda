#pragma once

#include "backends/cuda/bitplane_layout.hpp"

// The kernels of the bitplane search. The variants that were measured against
// them live in experiments/rng-reduction/diagnostic_kernels.cuh. The kernels
// are static because the search and the harness each compile this header.
namespace cuda_bitplanes {

// Hashes one field word: 64 x positions on one hashed z row of one y level.
// The bits are gathered 32 at a time so the shifts stay in 32-bit registers.
static __global__ void fill_planes(Layout layout, const int* plane_y, size_t count, Planes* planes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const int word = index % layout.field_words;
  const size_t row = index / layout.field_words;
  const int z_index = row % layout.hashed_rows;
  const int y = plane_y[row / layout.hashed_rows];
  const int64_t z = layout.first_hashed_z + static_cast<int64_t>(z_index) * STRIDE;
  const int64_t x_offset = static_cast<int64_t>(word) * WORD_BITS;
  const int64_t remaining = layout.field_width - x_offset;
  const int bits = remaining < WORD_BITS ? static_cast<int>(remaining) : WORD_BITS;
  const int64_t z_term = z * 116129781LL;
  uint32_t x_term = static_cast<uint32_t>(layout.field_x + x_offset) * 3129871u;

  Planes pair{};
  if (bits > 0) {
    for (int half = 0; half < 2; ++half) {
      uint32_t low = 0, high = 0;
      for (int bit = 0; bit < 32; ++bit, x_term += 3129871u) {
        const uint32_t rotation = modern_from_coord_random(coord_random_from_parts(x_term, z_term, y), 4);
        low = (low >> 1) | (rotation << 31);
        high = (high >> 1) | ((rotation >> 1) << 31);
      }
      pair.low |= static_cast<Word>(low) << (half * 32);
      pair.high |= static_cast<Word>(high) << (half * 32);
    }
    pair.low &= low_mask(bits);
    pair.high &= low_mask(bits);
  }
  planes[index] = pair;
}

// The candidates of one word that pass every plane check of their z coset.
// `y` and `z_row` count from the slab's first level and first row.
__device__ __forceinline__ Word filter_word(const Layout& layout, const PlaneCheck* checks,
                                            const int* coset_offsets, const Planes* planes, size_t y,
                                            size_t z_row, int word) {
  const int remaining = layout.nx - word * WORD_BITS;
  Word mask = low_mask(remaining < WORD_BITS ? remaining : WORD_BITS);
  // The coset anchor is within the slab's first STRIDE rows.
  // This quotient is the number of strides since that anchor.
  const size_t base = (y * layout.hashed_rows + z_row / STRIDE) * layout.field_words + word;
  const int coset = residue(layout.bounds.z_min + static_cast<int>(z_row));
  for (int i = coset_offsets[coset]; i < coset_offsets[coset + 1] && mask; ++i) {
    const auto check = checks[i];
    const size_t index = base + check.offset;
    Word low = planes[index].low >> check.shift;
    if (check.shift)
      low |= planes[index + 1].low << (WORD_BITS - check.shift);
    mask &= (check.rotation & 1) ? low : ~low;
    if (!check.is_side) {
      Word high = planes[index].high >> check.shift;
      if (check.shift)
        high |= planes[index + 1].high << (WORD_BITS - check.shift);
      mask &= (check.rotation & 2) ? high : ~high;
    }
  }
  return mask;
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
    const uint32_t x_term = static_cast<uint32_t>(x) * 3129871u;
    bool passes = true;
    for (int i = verify_offsets[coset]; i < verify_offsets[coset + 1]; ++i) {
      const VerifyCheck check = verify[i];
      const int rotation = modern_from_coord_random(
          coord_random_from_parts(x_term + check.x_term, z_term + check.z_term, candidate_y + check.y_offset),
          4);
      if ((rotation & check.mask) != check.rotation) {
        passes = false;
        break;
      }
    }
    if (passes) {
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
