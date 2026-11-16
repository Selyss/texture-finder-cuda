#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "bitplanes/plan.hpp"
#include "search/backend.hpp"

namespace bitplanes {

// run_shared_search for a backend with a bitplane search. The directions that
// can use bitplanes go through `pass` together when there are at least two,
// so they share one field. `pass` takes their formations and returns an
// optional SharedSearchResult, empty if it cannot plan them. Every other
// direction is searched alone by run_search.
template <class Pass>
[[nodiscard]] SharedSearchResult search_sharing_field(const SearchBounds& bounds,
                                                      std::span<const std::vector<BlockInfo>> directions,
                                                      Version version, Pass&& pass) {
  std::vector<std::size_t> shared_indices;
  std::vector<std::vector<BlockInfo>> shared_blocks;
  if (shared_search_enabled(version)) {
    for (std::size_t i{0}; i < directions.size(); ++i) {
      if (!directions[i].empty() && eligible(directions[i])) {
        shared_indices.push_back(i);
        shared_blocks.push_back(directions[i]);
      }
    }
  }

  SharedSearchResult result;
  result.directions.resize(directions.size());
  std::vector<bool> done(directions.size(), false);
  if (shared_blocks.size() > 1) {
    if (std::optional<SharedSearchResult> shared{
            pass(std::span<const std::vector<BlockInfo>>{shared_blocks})}) {
      result.kernel_seconds += shared->kernel_seconds;
      for (std::size_t i{0}; i < shared_indices.size(); ++i) {
        result.directions[shared_indices[i]] = std::move(shared->directions[i]);
        done[shared_indices[i]] = true;
      }
    }
  }
  for (std::size_t i{0}; i < directions.size(); ++i) {
    if (done[i])
      continue;
    result.directions[i] = run_search(bounds, directions[i], version);
    result.kernel_seconds += result.directions[i].kernel_seconds;
  }
  return result;
}

} // namespace bitplanes
