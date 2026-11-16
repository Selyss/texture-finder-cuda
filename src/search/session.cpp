#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#define TF_ISATTY_STDERR() _isatty(_fileno(stderr))
#else
#include <unistd.h>
#define TF_ISATTY_STDERR() isatty(fileno(stderr))
#endif

#include "session.hpp"
#include "checkpoint.hpp"
#include "texture/rotation.hpp"

namespace {

// Volumes above this are searched in z-band slices so the driver can report
// progress, stream matches, and checkpoint between slices.
constexpr unsigned long long SLICE_THRESHOLD = 1'000'000'000ull;
constexpr unsigned long long FIRST_SLICE_POSITIONS = 200'000'000ull;
constexpr double TARGET_SLICE_SECONDS = 2.0;

const char* DIRECTION_NAMES[4] = {"North", "West", "South", "East"};

struct Progress {
  unsigned long long done = 0;
  unsigned long long total = 0;
  bool enabled = false;
  bool tty = false;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point last_print;
  bool line_pending = false;

  void begin(unsigned long long total_positions, unsigned long long already_done) {
    total = total_positions;
    done = already_done;
    enabled = true;
    tty = TF_ISATTY_STDERR() != 0;
    start = last_print = std::chrono::steady_clock::now();
  }

  void update(unsigned long long positions) {
    if (!enabled)
      return;
    done += positions;
    const auto now = std::chrono::steady_clock::now();
    const double since_last = std::chrono::duration<double>(now - last_print).count();
    if (since_last < (tty ? 0.2 : 5.0) && done < total)
      return;
    last_print = now;

    const double elapsed = std::chrono::duration<double>(now - start).count();
    const double rate = elapsed > 0 ? (double)done / elapsed : 0;
    const double pct = total > 0 ? 100.0 * (double)done / (double)total : 100.0;
    char eta[64] = "--:--:--";
    if (rate > 0 && done < total) {
      long long s = (long long)((double)(total - done) / rate);
      std::snprintf(eta, sizeof(eta), "%02lld:%02lld:%02lld", s / 3600, s / 60 % 60, s % 60);
    }
    char line[160];
    std::snprintf(line, sizeof(line), "Progress: %5.1f%% | %.3g/%.3g positions | %.3g/s | ETA %s", pct,
                  (double)done, (double)total, rate, eta);
    if (tty) {
      std::fprintf(stderr, "\r%-100s", line);
      line_pending = true;
    } else
      std::fprintf(stderr, "%s\n", line);
    std::fflush(stderr);
  }

  // Keeps streamed stdout matches from colliding with the \r status line.
  void clear_line() {
    if (enabled && tty && line_pending) {
      std::fprintf(stderr, "\r%-100s\r", "");
      std::fflush(stderr);
      line_pending = false;
    }
  }

  void end() {
    if (enabled && tty && line_pending)
      std::fprintf(stderr, "\n");
  }
};

void print_match(const MatchResult& m, int direction, bool all_mode) {
  std::cout << "Match found at [" << m.x << ", " << m.y << ", " << m.z << "]";
  if (all_mode)
    std::cout << " facing " << DIRECTION_NAMES[direction];
  std::cout << "\n";
}

} // namespace

SearchOutcome run_search_session(const SearchRequest& request) {
  const auto& bounds = request.bounds;
  const auto& formation = request.formation;
  const auto version = request.version;
  const auto& directions = request.directions;
  const auto& checkpoint_path = request.checkpoint_path;
  std::cout << "Backend: " << search_backend_name() << std::endl;
  const char* VERSION_NAMES[NUM_VERSIONS] = {"1.21.2+", "1.13 - 1.21.1", "<=1.12.2",
                                             "Sodium 1.0-4.1 (MC 1.16 - 1.18.2)",
                                             "Sodium 4.2-4.8 (MC 1.19 - 1.19.3)"};
  std::cout << "Version: " << VERSION_NAMES[version] << std::endl;
  const bool all_mode = directions.size() > 1;
  if (!all_mode)
    std::cout << "Facing: " << DIRECTION_NAMES[directions[0]] << std::endl;
  else
    std::cout << "Facing: trying all directions" << std::endl;

  std::vector<BlockInfo> tops_and_bottoms, sides;
  for (const auto& info : formation)
    (info.is_side ? sides : tops_and_bottoms).push_back(info);

  if (tops_and_bottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS) {
    std::cerr << "Error: formation has too many blocks (max " << MAX_FORMATION_BLOCKS << " per face type)"
              << std::endl;
    return {1};
  }

  const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
  const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
  const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
  if (nx > (~0ull) / ny || nx * ny > (~0ull) / nz) {
    std::cerr << "Error: search volume too large" << std::endl;
    return {1};
  }
  const unsigned long long rows_per_z = nx * ny;
  const double volume = (double)nx * (double)ny * (double)nz;

  // Uniqueness lint: each top block divides coincidental matches by 4,
  // each side block by 2. Report exactly how much constraint is missing.
  const double have_bits = 2.0 * (double)tops_and_bottoms.size() + (double)sides.size();
  const double need_bits = std::log2(volume) + 6.64; // expected <= ~0.01 coincidences
  const double expected_random = volume * std::pow(2.0, -have_bits);
  if (need_bits > have_bits) {
    const int more_tops = (int)std::ceil((need_bits - have_bits) / 2.0);
    std::cout << "Warning: formation is weak for this volume; ~" << expected_random
              << " coincidental matches expected. Add ~" << more_tops << " more top-face blocks (or "
              << 2 * more_tops << " side blocks) for a unique result." << std::endl;
  }

  Checkpoint ckpt;
  ckpt.slice_first = all_mode && shared_search_enabled(version);
  ckpt.next_z = bounds.z_min; // progress baseline even when no checkpoint is used
  if (!checkpoint_path.empty()) {
    ckpt.signature = checkpoint_signature(request);
    ckpt.path = checkpoint_path;
    ckpt.active = true;
    try {
      if (ckpt.load()) {
        if (ckpt.slice_first && (!all_mode || ckpt.dir_index != 0 || ckpt.next_z < bounds.z_min ||
                                 ckpt.next_z > static_cast<long long>(bounds.z_max) + 1))
          throw std::runtime_error("invalid slice-first checkpoint: " + ckpt.path);
        std::cout << "Resuming from checkpoint: direction index " << ckpt.dir_index << ", z = " << ckpt.next_z
                  << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return {1};
    }
  }

  // Debug/test hook: stop (exit 3) after N slices, leaving the checkpoint
  // behind, so resume can be exercised deterministically.
  long long max_slices = -1;
  if (const char* ms = std::getenv("TF_MAX_SLICES"))
    max_slices = std::atoll(ms);
  long long slices_done = 0;

  const unsigned long long volume_per_dir = rows_per_z * nz;
  const bool sliced = ckpt.active || volume_per_dir > SLICE_THRESHOLD;

  Accum acc = ckpt.acc;
  Progress progress;
  if (sliced)
    progress.begin(volume_per_dir * directions.size(),
                   (unsigned long long)ckpt.dir_index * volume_per_dir +
                       (unsigned long long)((long long)ckpt.next_z - bounds.z_min) * rows_per_z *
                           (ckpt.slice_first ? directions.size() : 1));

  // Adaptive slice sizing: aim for TARGET_SLICE_SECONDS per slice based on
  // the measured rate of the previous slice.
  double rate_ema = 0.0;

  struct Found {
    MatchResult m;
    int direction;
  };
  std::vector<Found> collected;

  const auto rotate = [&](int dir) {
    SearchFormation rotated{tops_and_bottoms, sides};
    for (int r = 0; r < dir; ++r) {
      for (auto& b : rotated.tops)
        rotate_formation(b);
      for (auto& b : rotated.sides)
        rotate_formation(b);
    }
    return rotated;
  };
  const auto accept = [&](const DirectionResult& result, int dir) {
    acc.truncated |= result.truncated;
    acc.total_matches += result.total;
    acc.listed += result.matches.size();
    if (!sliced) {
      for (const auto& match : result.matches)
        collected.push_back({match, dir});
    } else if (!result.matches.empty()) {
      progress.clear_line();
      for (const auto& match : result.matches)
        print_match(match, dir, all_mode);
      std::cout.flush();
    }
  };
  const auto slice_end = [&](long long z, unsigned long long positions_per_row) {
    const auto target = rate_ema > 0 ? static_cast<unsigned long long>(rate_ema * TARGET_SLICE_SECONDS)
                                     : FIRST_SLICE_POSITIONS;
    const auto rows = std::max(1ull, target / positions_per_row);
    return rows <= static_cast<unsigned long long>(bounds.z_max - z) ? z + static_cast<long long>(rows) - 1
                                                                     : bounds.z_max;
  };
  const auto record_slice = [&](unsigned long long positions, double seconds, size_t dir, long long next_z,
                                bool work_remains) {
    acc.kernel_sec += seconds;
    if (seconds > 0) {
      const double rate = static_cast<double>(positions) / seconds;
      rate_ema = rate_ema > 0 ? 0.5 * rate_ema + 0.5 * rate : rate;
    }
    progress.update(positions);
    ckpt.dir_index = dir;
    ckpt.next_z = next_z;
    ckpt.acc = acc;
    ckpt.save();

    ++slices_done;
    if (max_slices < 0 || slices_done < max_slices || !work_remains)
      return false;
    progress.end();
    std::cout << "Stopping after " << slices_done << " slice(s) (TF_MAX_SLICES)";
    if (ckpt.active)
      std::cout << "; state saved to " << ckpt.path;
    std::cout << std::endl;
    return true;
  };

  if (ckpt.slice_first) {
    std::vector<SearchFormation> formations;
    for (int dir : directions)
      formations.push_back(rotate(dir));
    const auto positions_per_row = rows_per_z * directions.size();
    for (long long z = ckpt.next_z; z <= bounds.z_max;) {
      SearchBounds slice = bounds;
      const auto end = sliced ? slice_end(z, positions_per_row) : bounds.z_max;
      slice.z_min = static_cast<int>(z);
      slice.z_max = static_cast<int>(end);
      const auto batch = run_shared_search(slice, formations, version);
      for (size_t di = 0; di < batch.directions.size(); ++di)
        accept(batch.directions[di], directions[di]);
      if (sliced) {
        if (record_slice((end - z + 1) * positions_per_row, batch.kernel_seconds, 0, end + 1,
                         end < bounds.z_max))
          return {3};
      } else
        acc.kernel_sec += batch.kernel_seconds;
      z = end + 1;
    }
  }

  for (size_t di = ckpt.dir_index; !ckpt.slice_first && di < directions.size(); ++di) {
    const int dir = directions[di];
    const auto rotated = rotate(dir);
    const auto& tops = rotated.tops;
    const auto& sids = rotated.sides;

    if (!sliced) {
      const auto result = run_search(bounds, tops, sids, version);
      acc.kernel_sec += result.kernel_ms / 1000.0;
      accept(result, dir);
      continue;
    }

    long long z = (di == ckpt.dir_index && ckpt.resumed) ? ckpt.next_z : bounds.z_min;
    while (z <= bounds.z_max) {
      const auto end = slice_end(z, rows_per_z);
      SearchBounds sub = bounds;
      sub.z_min = static_cast<int>(z);
      sub.z_max = static_cast<int>(end);
      const auto result = run_search(sub, tops, sids, version);
      accept(result, dir);

      const auto positions = static_cast<unsigned long long>(end - z + 1) * rows_per_z;
      z = end + 1;
      if (record_slice(positions, result.kernel_ms / 1000.0, di, z,
                       z <= bounds.z_max || di + 1 < directions.size()))
        return {3};
    }
    ckpt.dir_index = di + 1;
    ckpt.next_z = bounds.z_min;
    ckpt.acc = acc;
    ckpt.save();
  }

  progress.end();

  if (!sliced) {
    std::sort(collected.begin(), collected.end(), [](const Found& a, const Found& b) {
      return a.m.x != b.m.x   ? a.m.x < b.m.x
             : a.m.y != b.m.y ? a.m.y < b.m.y
             : a.m.z != b.m.z ? a.m.z < b.m.z
                              : a.direction < b.direction;
    });
    for (const auto& f : collected)
      print_match(f.m, f.direction, all_mode);
  }

  if (acc.total_matches == 0)
    std::cout << "No matches found." << std::endl;
  else
    std::cout << acc.total_matches << (acc.total_matches == 1 ? " match" : " matches") << std::endl;
  if (acc.truncated)
    std::cout << "Warning: only the first " << acc.listed << " of " << acc.total_matches
              << " matches are listed; shrink the search area." << std::endl;

  ckpt.finish();

  return {0, acc.kernel_sec};
}
