#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct SearchRequest;

struct Totals {
  std::uint64_t matches{0};
  std::uint64_t listed{0};
  double kernel_seconds{0.0};
  bool truncated{false};
};

// Where an interrupted search resumes: the direction in flight, the next
// unsearched z, and the totals so far. The signature covers the search
// arguments and the formation, so a file from another search is rejected.
struct Checkpoint {
  std::string path; // empty: checkpointing is off
  std::uint64_t signature{0};
  bool slice_first{false}; // each z slice covers every direction ("tfcp2")
  bool resumed{false};
  std::size_t direction_index{0};
  std::int64_t next_z{0};
  Totals totals;

  [[nodiscard]] bool active() const noexcept { return !path.empty(); }

  void save() const;
  void remove() const;

  // False if there is no file. Throws if the file is corrupt or belongs to
  // a different search.
  [[nodiscard]] bool load();
};

[[nodiscard]] std::uint64_t checkpoint_signature(const SearchRequest& request);
