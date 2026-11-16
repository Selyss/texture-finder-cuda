#pragma once

#include <algorithm>
#include <span>
#include <tuple>
#include <vector>

#include "search/backend.hpp"

// What one thread found for one direction, and how the threads' finds become
// a DirectionResult.
struct WorkerMatches {
  std::vector<MatchResult> matches; // at most MAX_MATCHES
  std::uint64_t total{0};

  void add(MatchResult match) {
    ++total;
    if (matches.size() < MAX_MATCHES)
      matches.push_back(match);
  }
};

template <class Workers, class Select>
[[nodiscard]] DirectionResult collect_matches(const Workers& workers, Select select) {
  DirectionResult result;
  for (const auto& worker : workers) {
    const WorkerMatches& found{select(worker)};
    result.total += found.total;
    const auto room{MAX_MATCHES - result.matches.size()};
    const auto take{std::min(room, found.matches.size())};
    result.matches.insert(result.matches.end(), found.matches.begin(), found.matches.begin() + take);
  }
  result.truncated = result.total > MAX_MATCHES;
  std::ranges::sort(result.matches, {}, [](const MatchResult& m) { return std::tuple{m.x, m.y, m.z}; });
  return result;
}
