#pragma once

#include <chrono>
#include <cstdint>

// The progress line on stderr: redrawn in place on a terminal, appended every
// few seconds when stderr is a file or pipe. Silent until start() is called.
class Progress {
public:
  void start(std::uint64_t total_positions, std::uint64_t already_done);
  void advance(std::uint64_t positions);

  // Call before writing matches to stdout so they do not land on the status line.
  void clear_line();
  void finish();

private:
  using Clock = std::chrono::steady_clock;

  std::uint64_t done_{0};
  std::uint64_t total_{0};
  bool enabled_{false};
  bool terminal_{false};
  bool line_pending_{false};
  Clock::time_point started_;
  Clock::time_point last_print_;
};
