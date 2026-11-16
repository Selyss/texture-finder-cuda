#pragma once

#include <string>
#include <vector>

#include "search/backend.hpp"

// A parsed CLI request. Directions and formation order also identify a checkpoint.
struct SearchRequest {
  SearchBounds bounds;
  const std::vector<BlockInfo>& formation;
  int version;
  const std::vector<int>& directions;
  const std::string& checkpoint_path;
};

struct SearchOutcome {
  int exit_status;
  double kernel_seconds = 0.0;
};

// Runs the search, streams matches and manages checkpoints.
SearchOutcome run_search_session(const SearchRequest& request);
