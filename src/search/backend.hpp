#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "texture/formation.hpp"

// The interface every backend implements. The build links exactly one of
// src/backends/{cpu,cuda,metal}; all three share src/texture/texture_rules.hpp
// and must return identical results (test/e2e.sh checks the linked one).

// Inclusive search bounds.
struct SearchBounds {
  int x_min, x_max;
  int y_min, y_max;
  int z_min, z_max;

  [[nodiscard]] constexpr std::uint64_t nx() const noexcept { return count(x_min, x_max); }
  [[nodiscard]] constexpr std::uint64_t ny() const noexcept { return count(y_min, y_max); }
  [[nodiscard]] constexpr std::uint64_t nz() const noexcept { return count(z_min, z_max); }

private:
  [[nodiscard]] static constexpr std::uint64_t count(int min, int max) noexcept {
    return static_cast<std::uint64_t>(std::int64_t{max} - min + 1);
  }
};

struct MatchResult {
  int x, y, z;
};

// A search that produces more matches than this has a formation far too small
// to identify a location. Backends count every match but store only this many.
constexpr unsigned int MAX_MATCHES{1u << 20};

struct DirectionResult {
  std::vector<MatchResult> matches; // sorted by x, y, z
  std::uint64_t total{0};           // includes matches beyond MAX_MATCHES
  bool truncated{false};
  double kernel_seconds{0.0};
};

struct SharedSearchResult {
  std::vector<DirectionResult> directions;
  double kernel_seconds{0.0};
};

// "CPU", "CUDA" or "Metal".
[[nodiscard]] std::string_view search_backend_name();

// `blocks` is one direction's formation and must not be empty. Checks run in
// the order given, so the session lists top faces first: they reject more.
[[nodiscard]] DirectionResult run_search(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks,
                                         int version);

// True when the backend can search several directions in one pass.
[[nodiscard]] bool shared_search_enabled(int version);
[[nodiscard]] SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                                   std::span<const std::vector<BlockInfo>> directions,
                                                   int version);

// run_shared_search for a backend with no shared pass: one direction at a time.
[[nodiscard]] inline SharedSearchResult
search_each_direction(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> directions,
                      int version) {
  SharedSearchResult result;
  for (const auto& blocks : directions) {
    result.directions.push_back(run_search(bounds, blocks, version));
    result.kernel_seconds += result.directions.back().kernel_seconds;
  }
  return result;
}
