#pragma once

#include <type_traits>

#include "backends/cuda/bitplanes.cuh"

// Kernel variants the harness times against the production bitplane kernels.
// The filter and the survivor check of the precomputed variants are the
// production functions, so the harness measures and checks the shipped code.
namespace cuda_bitplanes::diagnostic {

// Full      filter, survivor check, masks and counts
// Filter    filter, masks and counts
// NoMemory  Filter on synthetic words instead of the field
// Div32     Filter with 32-bit row arithmetic (unprecomputed checks only)
// NoTail    filter alone, nothing stored
// Search    filter and survivor check, as production runs it
enum class MatchMode { Full, Filter, NoMemory, Div32, NoTail, Search };

// The first fill kernel: one rotation bit per iteration. With WorldY false the
// field repeats every block y offset for each candidate y.
template <bool WorldY>
__global__ void fill_planes_plain(Layout layout, const int* y_offsets, size_t count, Planes* planes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const int word = index % layout.field_words;
  const size_t row = index / layout.field_words;
  const int z_index = row % layout.hashed_rows;
  const size_t level = row / layout.hashed_rows;
  const int y = WorldY ? y_offsets[level]
                       : layout.bounds.y_min + static_cast<int>(level / layout.y_slots) +
                             y_offsets[level % layout.y_slots];
  const int64_t z = layout.first_hashed_z + static_cast<int64_t>(z_index) * STRIDE;
  const int64_t x_offset = static_cast<int64_t>(word) * WORD_BITS;
  const int64_t remaining = layout.field_width - x_offset;
  const int bits = remaining < WORD_BITS ? static_cast<int>(remaining) : WORD_BITS;
  const int64_t z_term = z * 116129781LL;
  uint32_t x_term = static_cast<uint32_t>(layout.field_x + x_offset) * 3129871u;

  Planes pair{};
  for (int bit = 0; bit < bits; ++bit, x_term += 3129871u) {
    const unsigned int rotation = modern_from_coord_random(coord_random_from_parts(x_term, z_term, y), 4);
    pair.low |= static_cast<Word>(rotation & 1u) << bit;
    pair.high |= static_cast<Word>(rotation >> 1) << bit;
  }
  planes[index] = pair;
}

__device__ inline unsigned int warp_sum(unsigned int count) {
  for (int step = 16; step > 0; step /= 2)
    count += __shfl_down_sync(0xffffffffu, count, step);
  return count;
}

__device__ inline Word synthetic_word(size_t index, unsigned int salt) {
  // Mix adjacent addresses and give each check and plane its own stream.
  Word value = index ^ (static_cast<Word>(salt) << 32);
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

// Precomputed checks run the production filter. Unprecomputed checks keep the
// earlier form, which locates each field word inside the kernel.
template <MatchMode Mode, bool WorldY, bool Precomputed = false>
__global__ void
match_words_probe(Layout layout, const std::conditional_t<Precomputed, PlaneCheck, Check>* checks,
                  const int* coset_offsets, const Planes* planes, unsigned int* block_counts, Word* masks,
                  int mask_z_rows, const VerifyCheck* verify, const int* verify_offsets, MatchResult* matches,
                  Word* total_matches, Word capacity) {
  const int segments = ceil_div(layout.candidate_words, THREADS);
  using Row = std::conditional_t<Mode == MatchMode::Div32, unsigned int, size_t>;
  const Row row = blockIdx.x / segments;
  const int word = (blockIdx.x % segments) * THREADS + threadIdx.x;
  const Row y = row / layout.nz;
  const Row z_row = row % layout.nz;
  Word mask = 0;
  if (word < layout.candidate_words) {
    if constexpr (Precomputed) {
      static_assert(WorldY, "Precomputed offsets use compact world-y levels");
      mask = filter_word(layout, checks, coset_offsets, planes, y, z_row, word);
    } else {
      const int z = layout.bounds.z_min + static_cast<int>(z_row);
      const int coset = residue(z);
      const int remaining = layout.nx - word * WORD_BITS;
      mask = low_mask(remaining < WORD_BITS ? remaining : WORD_BITS);
      for (int i = coset_offsets[coset]; i < coset_offsets[coset + 1] && mask; ++i) {
        const auto check = checks[i];
        const size_t z_index = (static_cast<int64_t>(z) + check.z_offset - layout.first_hashed_z) / STRIDE;
        const size_t level = WorldY ? static_cast<size_t>(check.y_slot) + y
                                    : static_cast<size_t>(y) * layout.y_slots + check.y_slot;
        const size_t field_row = level * layout.hashed_rows + z_index;
        const size_t index = field_row * layout.field_words + word + check.x_offset / WORD_BITS;
        const int shift = check.x_offset % WORD_BITS;
        Word low = (Mode == MatchMode::NoMemory ? synthetic_word(index, 2 * i) : planes[index].low) >> shift;
        if (shift)
          low |= (Mode == MatchMode::NoMemory ? synthetic_word(index + 1, 2 * i) : planes[index + 1].low)
                 << (WORD_BITS - shift);
        mask &= (check.rotation & 1) ? low : ~low;
        if (!check.is_side) {
          Word high =
              (Mode == MatchMode::NoMemory ? synthetic_word(index, 2 * i + 1) : planes[index].high) >> shift;
          if (shift)
            high |=
                (Mode == MatchMode::NoMemory ? synthetic_word(index + 1, 2 * i + 1) : planes[index + 1].high)
                << (WORD_BITS - shift);
          mask &= (check.rotation & 2) ? high : ~high;
        }
      }
    }
    if (masks)
      masks[(y * mask_z_rows + z_row) * layout.candidate_words + word] = mask;
  }

  if constexpr (Mode == MatchMode::Full || Mode == MatchMode::Search)
    verify_survivors<Version::modern>(layout, mask, y, z_row, word, verify, verify_offsets, matches,
                                      total_matches, capacity);

  if constexpr (Mode == MatchMode::NoTail) {
    // Keep every thread's filter work live without storing masks during timing.
    asm volatile("" ::"l"(mask));
  } else if constexpr (Mode != MatchMode::Search) {
    __shared__ unsigned int warp_counts[THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const unsigned int count = warp_sum(__popcll(mask));
    if (!lane)
      warp_counts[warp] = count;
    __syncthreads();
    if (!warp) {
      const unsigned int total = warp_sum(lane < THREADS / 32 ? warp_counts[lane] : 0);
      if (!lane)
        block_counts[blockIdx.x] = total;
    }
  }
}

} // namespace cuda_bitplanes::diagnostic
