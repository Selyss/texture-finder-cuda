#pragma once

#include <optional>
#include <span>
#include <vector>

#include "search/backend.hpp"

// The bitplane search on CPU threads: each thread hashes the needed z rows of
// a small tile once into a field of rotation bits, then tests 64 candidates
// per word against it. Works for every version.
namespace cpu_bitplanes {

// Searches every formation against one shared field per tile. nullopt if the
// search cannot be planned, in which case the caller uses the plain scan.
[[nodiscard]] std::optional<SharedSearchResult>
search(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations, Version version);

} // namespace cpu_bitplanes
