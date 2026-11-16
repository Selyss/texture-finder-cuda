#include "search/progress.hpp"

#include <array>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using Seconds = std::chrono::duration<double>;

constexpr Seconds TERMINAL_INTERVAL{0.2};
constexpr Seconds LOG_INTERVAL{5.0};

[[nodiscard]] bool stderr_is_terminal() {
#ifdef _WIN32
  return _isatty(_fileno(stderr)) != 0;
#else
  return isatty(fileno(stderr)) != 0;
#endif
}

} // namespace

void Progress::start(std::uint64_t total_positions, std::uint64_t already_done) {
  total_ = total_positions;
  done_ = already_done;
  enabled_ = true;
  terminal_ = stderr_is_terminal();
  started_ = last_print_ = Clock::now();
}

void Progress::advance(std::uint64_t positions) {
  if (!enabled_)
    return;
  done_ += positions;
  const auto now{Clock::now()};
  if (now - last_print_ < (terminal_ ? TERMINAL_INTERVAL : LOG_INTERVAL) && done_ < total_)
    return;
  last_print_ = now;

  const auto done{static_cast<double>(done_)};
  const auto total{static_cast<double>(total_)};
  const auto elapsed{Seconds{now - started_}.count()};
  const auto rate{elapsed > 0 ? done / elapsed : 0.0};
  const auto percent{total_ > 0 ? 100.0 * done / total : 100.0};

  std::array<char, 64> eta{"--:--:--"};
  if (rate > 0 && done_ < total_) {
    const auto left{static_cast<long long>((total - done) / rate)};
    std::snprintf(eta.data(), eta.size(), "%02lld:%02lld:%02lld", left / 3600, left / 60 % 60, left % 60);
  }
  std::array<char, 160> line{};
  std::snprintf(line.data(), line.size(), "Progress: %5.1f%% | %.3g/%.3g positions | %.3g/s | ETA %s",
                percent, done, total, rate, eta.data());

  if (terminal_) {
    std::fprintf(stderr, "\r%-100s", line.data());
    line_pending_ = true;
  } else {
    std::fprintf(stderr, "%s\n", line.data());
  }
  std::fflush(stderr);
}

void Progress::clear_line() {
  if (!line_pending_)
    return;
  std::fprintf(stderr, "\r%-100s\r", "");
  std::fflush(stderr);
  line_pending_ = false;
}

void Progress::finish() {
  if (line_pending_)
    std::fprintf(stderr, "\n");
}
