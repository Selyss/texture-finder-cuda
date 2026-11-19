#pragma once

#include <optional>
#include <span>
#include <vector>

#include "search/backend.hpp"

// The bitplane search on Metal: hash each needed z row once into a field of
// rotation bits, then test 64 candidates per word against it. Works for
// every version.
namespace metal_bitplanes {

// Searches every formation against one shared field. nullopt if the search
// cannot be planned, in which case the caller uses the plain kernel.
[[nodiscard]] std::optional<SharedSearchResult>
search(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations, Version version);

} // namespace metal_bitplanes
