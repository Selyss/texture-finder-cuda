#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "backends/cuda/bitplane_layout.hpp"

// Decides how a search is cut into slabs and which blocks each kernel checks.
// Host-only and free of CUDA calls, so the unit tests cover it.
namespace cuda_bitplanes {

// One direction's blocks grouped by z coset. A candidate on coset c reads the
// plane checks in [offsets[c], offsets[c + 1]) from the field. Its other
// blocks fall between hashed rows and are hashed per survivor instead.
struct DirectionPlan {
  std::vector<Check> checks;
  std::vector<VerifyCheck> verify;
  std::array<int, STRIDE + 1> offsets{}, verify_offsets{};
};

// One launch of the fill and match kernels.
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
// if the search cannot use bitplanes within `plane_bytes` of field memory.
[[nodiscard]] std::optional<SearchPlan> plan_search(const SearchBounds& bounds,
                                                    std::span<const std::vector<BlockInfo>> formations,
                                                    std::size_t plane_bytes);

} // namespace cuda_bitplanes
