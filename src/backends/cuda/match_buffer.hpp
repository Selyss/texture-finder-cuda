#pragma once

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "backends/cuda/device.hpp"
#include "search/backend.hpp"

namespace cuda {

// Where a kernel reports matches: a counter it bumps with atomicAdd for every
// match, and room for the first MAX_MATCHES of them.
class MatchBuffer {
public:
  using Count = unsigned long long;

  MatchBuffer()
      : matches_{MAX_MATCHES}
      , count_{1} {
    count_.zero();
  }

  [[nodiscard]] MatchResult* matches() const noexcept { return matches_.get(); }
  [[nodiscard]] Count* count() const noexcept { return count_.get(); }

  // Copies the matches back, sorted by x, y, z. The time is left for the caller.
  [[nodiscard]] DirectionResult collect() const {
    const auto total{count_.download(1).front()};
    auto matches{matches_.download(std::min<Count>(total, MAX_MATCHES))};
    std::sort(matches.begin(), matches.end(), [](const MatchResult& a, const MatchResult& b) {
      return std::tuple{a.x, a.y, a.z} < std::tuple{b.x, b.y, b.z};
    });
    return {.matches = std::move(matches), .total = total, .truncated = total > MAX_MATCHES};
  }

private:
  DeviceArray<MatchResult> matches_;
  DeviceArray<Count> count_;
};

// One result from the results of disjoint parts of a search. The time is
// left for the caller.
[[nodiscard]] inline DirectionResult merge_results(std::vector<DirectionResult> parts) {
  DirectionResult merged;
  for (auto& part : parts) {
    merged.total += part.total;
    merged.matches.insert(merged.matches.end(), part.matches.begin(), part.matches.end());
  }
  std::sort(merged.matches.begin(), merged.matches.end(), [](const MatchResult& a, const MatchResult& b) {
    return std::tuple{a.x, a.y, a.z} < std::tuple{b.x, b.y, b.z};
  });
  if (merged.matches.size() > MAX_MATCHES)
    merged.matches.resize(MAX_MATCHES);
  merged.truncated = merged.total > MAX_MATCHES;
  return merged;
}

} // namespace cuda
