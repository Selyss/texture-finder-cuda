#pragma once

#include <optional>
#include <span>
#include <vector>

#include "search/backend.hpp"

// The bitplane search on CPU threads: each thread hashes the needed z rows of
// a small tile once into a field of rotation bits, then tests 64 candidates
// per word against it. Works for every version.
namespace cpu_bitplanes {

struct SearchOutput {
  std::vector<DirectionResult> directions; // one per formation, times left at zero
  double seconds{0.0};                     // for the whole pass
};

// On unless TF_CPU_BITPLANES=0, which keeps every search on the plain scan.
[[nodiscard]] bool enabled();

// TF_CPU_BITPLANES_TRACE makes the backend say on stderr which path ran.
[[nodiscard]] bool trace_enabled();

// Searches every formation against one shared field per tile. nullopt if the
// search cannot be planned, in which case the caller uses the plain scan.
[[nodiscard]] std::optional<SearchOutput> search(const SearchBounds& bounds,
                                                 std::span<const std::vector<BlockInfo>> formations,
                                                 Version version);

} // namespace cpu_bitplanes
