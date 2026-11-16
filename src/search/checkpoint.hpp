#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct SearchRequest;

struct Accum {
  unsigned long long total_matches = 0;
  unsigned long long listed = 0;
  double kernel_sec = 0.0;
  bool truncated = false;
};

// Checkpoint state: which direction index is in flight, the next unsearched
// z within it, and the accumulated counters. Guarded by a signature over the
// search arguments and the formation file content so a stale or mismatched
// state file can never silently corrupt a different search.
struct Checkpoint {
  std::string path;
  bool active = false;
  bool resumed = false;
  bool slice_first = false;
  std::uint64_t signature = 0;
  std::size_t dir_index = 0;
  long long next_z = 0;
  Accum acc;

  void save() const;
  void finish() const;
  bool load();
};

std::uint64_t checkpoint_signature(const SearchRequest& request);
