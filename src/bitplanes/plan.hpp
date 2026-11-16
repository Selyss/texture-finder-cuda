#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "bitplanes/layout.hpp"

// Decides how a search is cut into slabs and which blocks each kernel checks.
// Plain host code shared by the backends, so the unit tests cover it.
namespace bitplanes {

// Locates each plane check's field word for one slab. Within a slab every
// candidate of a z coset reads its checks at the same offsets from its own
// position in the field, so the kernels add one base index per candidate.
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

// One direction's blocks grouped by z coset. A candidate on coset c reads the
// plane checks in [offsets[c], offsets[c + 1]) from the field. Its other
// blocks fall between hashed rows and are hashed per survivor instead.
struct DirectionPlan {
  std::vector<Check> checks;
  std::vector<VerifyCheck> verify;
  std::array<int, STRIDE + 1> offsets{}, verify_offsets{};
};

// One pass of the fill and the match over a range of z rows.
struct Slab {
  Layout layout;
  std::size_t field_words; // words the fill kernel writes
  std::size_t match_blocks;
};

struct SearchPlan {
  SearchBounds bounds;
  int min_x, min_z, max_z; // block offset extents over all directions
  int field_x, field_width, field_words, candidate_words;
  std::vector<int> plane_y; // world y of each field level, ascending
  int slab_rows;
  std::size_t plane_words; // field allocation, enough for any slab
  std::size_t match_blocks_per_row;
  std::vector<DirectionPlan> directions;

  // The slab whose first candidate row is bounds.z_min + z_offset.
  [[nodiscard]] Slab slab(std::int64_t z_offset) const;
};

// False when some z coset has too few filter bits, which would send most of
// its candidates through per-survivor verification.
[[nodiscard]] bool eligible(std::span<const BlockInfo> blocks);

// Each formation holds one direction's blocks and must not be empty. nullopt
// if the search cannot use bitplanes within `plane_bytes` of field memory. A
// slab never has more than `max_slab_rows` z rows.
[[nodiscard]] std::optional<SearchPlan> plan_search(const SearchBounds& bounds,
                                                    std::span<const std::vector<BlockInfo>> formations,
                                                    std::size_t plane_bytes,
                                                    std::int64_t max_slab_rows = DEFAULT_SLAB_ROWS);

} // namespace bitplanes
