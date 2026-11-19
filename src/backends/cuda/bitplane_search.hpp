#pragma once

#include <optional>
#include <span>
#include <vector>

#include "search/backend.hpp"

// The bitplane search: hash each needed z row once
// into a field of rotation bits, then test 64 candidates per word against it.
namespace cuda_bitplanes {

// Searches every formation against one shared field. nullopt if the field
// does not fit in memory, in which case the caller uses the standard kernel.
[[nodiscard]] std::optional<SharedSearchResult>
search(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations, Version version);

} // namespace cuda_bitplanes
