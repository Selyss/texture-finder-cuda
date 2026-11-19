#pragma once

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "bitplanes/plan.hpp"
#include "search/backend.hpp"

namespace bitplanes {

// A backend's bitplane search is on unless its variable is set to 0.
[[nodiscard]] inline bool enabled_by(const char* variable) {
  const auto* setting{std::getenv(variable)};
  return !setting || std::string_view{setting} != "0";
}

// run_search for a backend with a bitplane search. `pass` searches a list of
// formations with bitplanes and returns an optional SharedSearchResult, empty
// if it cannot plan them. `plain` is the backend's search of every position.
template <class Pass, class Plain>
[[nodiscard]] DirectionResult search_one(const std::vector<BlockInfo>& blocks, Version version, Pass&& pass,
                                         Plain&& plain) {
  if (shared_search_enabled(version) && eligible(blocks)) {
    if (std::optional<SharedSearchResult> found{pass(std::span<const std::vector<BlockInfo>>{&blocks, 1})}) {
      auto result{std::move(found->directions.front())};
      result.kernel_seconds = found->kernel_seconds;
      result.bitplanes = true;
      return result;
    }
  }
  return plain();
}

// run_shared_search for a backend with a bitplane search. The directions that
// can use bitplanes go through `pass` together when there are at least two,
// so they share one field. Every other direction is searched alone by
// run_search.
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
        result.directions[shared_indices[i]].bitplanes = true;
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
