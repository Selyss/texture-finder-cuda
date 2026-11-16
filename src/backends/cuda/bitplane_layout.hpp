#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "search/backend.hpp"
#include "texture/texture_rules.hpp"

// How the bitplane field is laid out, shared by the planner (host), the
// kernels (device) and the tests. The field stores the two rotation bits of
// every position on one z row in STRIDE, one 64-bit word pair per 64 x values.

namespace cuda_bitplanes {
constexpr int STRIDE = 5;
constexpr int WORD_BITS = 64;
constexpr int THREADS = 256;
constexpr int DEFAULT_SLAB_ROWS = 4096;
using Word = unsigned long long;

struct Planes {
  Word low, high;
};

struct Check {
  int x_offset, y_slot, z_offset, rotation;
  bool is_side;
};

struct PlaneCheck {
  size_t offset;
  int shift, rotation;
  bool is_side;
};

struct VerifyCheck {
  int64_t z_term;
  uint32_t x_term;
  int y_offset, rotation, mask;
};

struct Layout {
  SearchBounds bounds;
  int nx, ny, nz;
  int field_x, field_width, field_words, candidate_words;
  int y_slots, hashed_rows;
  int64_t first_hashed_z;
};

TF_HOST_DEVICE inline int residue(int64_t z) {
  const int r = static_cast<int>(z % STRIDE);
  return r < 0 ? r + STRIDE : r;
}

inline void prepare_plane_checks(const Layout& slab, const std::vector<Check>& checks,
                                 const int* coset_offsets, std::vector<PlaneCheck>& result) {
  for (int coset = 0; coset < STRIDE; ++coset) {
    const int64_t start = slab.bounds.z_min;
    const int64_t anchor = start + residue(coset - start);
    if (anchor > slab.bounds.z_max)
      continue;
    for (int i = coset_offsets[coset]; i < coset_offsets[coset + 1]; ++i) {
      const auto& check = checks[i];
      const size_t z_index = (anchor + check.z_offset - slab.first_hashed_z) / STRIDE;
      const size_t level = check.y_slot;
      result[i] = {(level * slab.hashed_rows + z_index) * slab.field_words + check.x_offset / WORD_BITS,
                   check.x_offset % WORD_BITS, check.rotation, check.is_side};
    }
  }
}

TF_HOST_DEVICE inline int ceil_div(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}

TF_HOST_DEVICE inline Word low_mask(int bits) {
  return bits == WORD_BITS ? ~0ull : (1ull << bits) - 1;
}

// Offsets are sorted. Each candidate interval maps to a consecutive run of indices.
inline std::vector<int> compact_levels(const SearchBounds& bounds, const std::vector<int>& y_offsets) {
  std::vector<int> plane_y;
  int64_t next_y = std::numeric_limits<int>::min();
  for (int dy : y_offsets) {
    const int64_t first = static_cast<int64_t>(bounds.y_min) + dy;
    const int64_t last = static_cast<int64_t>(bounds.y_max) + dy;
    for (int64_t level = std::max(next_y, first); level <= last; ++level)
      plane_y.push_back(static_cast<int>(level));
    next_y = last + 1;
  }
  return plane_y;
}

} // namespace cuda_bitplanes
