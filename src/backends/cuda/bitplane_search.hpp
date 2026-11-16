#pragma once

#include <optional>
#include <span>
#include <vector>

#include "search/backend.hpp"

// The bitplane search for the modern version: hash each needed z row once
// into a field of rotation bits, then test 64 candidates per word against it.
namespace cuda_bitplanes {

struct SearchOutput {
  std::vector<DirectionResult> directions; // one per formation, times left at zero
  double kernel_seconds{0.0};              // for the whole pass
};

// On unless TF_CUDA_BITPLANES=0, which keeps every search on the standard kernel.
[[nodiscard]] bool enabled();

// TF_CUDA_BITPLANES_TRACE makes the backend say on stderr which path ran.
[[nodiscard]] bool trace_enabled();

// Searches every formation against one shared field. nullopt if the field
// does not fit in memory, in which case the caller uses the standard kernel.
[[nodiscard]] std::optional<SearchOutput> search(const SearchBounds& bounds,
                                                 std::span<const std::vector<BlockInfo>> formations);

} // namespace cuda_bitplanes
