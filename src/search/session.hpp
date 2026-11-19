#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "search/backend.hpp"

// What goes to stdout. `text` is the full report. `quiet` is the match lines
// and nothing else, with no progress line. `json` prints nothing here: the
// caller writes one object from the SearchOutcome.
enum class OutputMode { text, quiet, json };

// One parsed command line. The formation is in file order, which is also what
// identifies a checkpoint.
struct SearchRequest {
  SearchBounds bounds;
  std::vector<BlockInfo> formation;
  Version version;
  std::vector<int> directions; // quarter turns to try, in order
  std::string checkpoint_path; // empty: no checkpoint
  OutputMode output{OutputMode::text};
};

enum class SessionStatus {
  finished,
  stopped, // TF_MAX_SLICES ended the run early
};

struct FoundMatch {
  MatchResult match;
  int direction; // quarter turns, as in SearchRequest::directions
};

struct SearchOutcome {
  SessionStatus status;
  double kernel_seconds{0.0};
  std::string search;             // which search ran: "bitplanes", "plain", or a mix
  std::uint64_t total_matches{0}; // includes matches beyond MAX_MATCHES
  bool truncated{false};
  std::vector<FoundMatch> matches; // json output only, sorted by x, y, z
};

// "North", "West", "South" or "East" for 0 to 3 quarter turns.
[[nodiscard]] std::string_view direction_name(int direction);

// Runs the search, prints matches and progress, and keeps the checkpoint
// current. Throws std::runtime_error for a request that cannot be searched.
[[nodiscard]] SearchOutcome run_search_session(const SearchRequest& request);
