#pragma once

#include <string>
#include <vector>

#include "search/backend.hpp"

// One parsed command line. The formation is in file order, which is also what
// identifies a checkpoint.
struct SearchRequest {
  SearchBounds bounds;
  std::vector<BlockInfo> formation;
  int version;
  std::vector<int> directions; // quarter turns to try, in order
  std::string checkpoint_path; // empty: no checkpoint
};

enum class SessionStatus {
  finished,
  stopped, // TF_MAX_SLICES ended the run early
};

struct SearchOutcome {
  SessionStatus status;
  double kernel_seconds{0.0};
};

// Runs the search, prints matches and progress, and keeps the checkpoint
// current. Throws std::runtime_error for a request that cannot be searched.
[[nodiscard]] SearchOutcome run_search_session(const SearchRequest& request);
