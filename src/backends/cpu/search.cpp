// Portable multithreaded CPU backend, and the readable reference for the GPU
// ones: it tests every position against every block with the shared texture
// rules. Threads take chunks of the linearised volume from an atomic cursor.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include "search/backend.hpp"
#include "texture/texture_rules.hpp"

std::string_view search_backend_name() {
  return "CPU";
}

namespace {

constexpr std::uint64_t CHUNK_POSITIONS{1u << 16};

struct Position {
  int x, y, z;
};

struct Job {
  SearchBounds bounds;
  std::span<const BlockInfo> blocks;
  int version;
  std::uint64_t positions;
  std::atomic<std::uint64_t> cursor{0};
};

struct WorkerResult {
  std::vector<MatchResult> matches; // at most MAX_MATCHES
  std::uint64_t found{0};
};

// Positions are numbered x fastest, then z, then y.
[[nodiscard]] Position position_at(const SearchBounds& bounds, std::uint64_t index) {
  const auto nx{bounds.nx()}, nz{bounds.nz()};
  return {.x = bounds.x_min + static_cast<int>(index % nx),
          .y = bounds.y_min + static_cast<int>(index / nx / nz),
          .z = bounds.z_min + static_cast<int>(index / nx % nz)};
}

void advance(Position& at, const SearchBounds& bounds) {
  if (++at.x <= bounds.x_max)
    return;
  at.x = bounds.x_min;
  if (++at.z <= bounds.z_max)
    return;
  at.z = bounds.z_min;
  ++at.y;
}

[[nodiscard]] bool formation_matches(std::span<const BlockInfo> blocks, int version, Position origin) {
  return std::ranges::all_of(blocks, [&](const BlockInfo& block) {
    const int mod{block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM};
    return get_texture_for_version(version, origin.x + block.x, origin.y + block.y, origin.z + block.z,
                                   mod) == block.rotation;
  });
}

void search_chunks(Job& job, WorkerResult& result) {
  for (;;) {
    const auto first{job.cursor.fetch_add(CHUNK_POSITIONS)};
    if (first >= job.positions)
      return;
    const auto end{std::min(first + CHUNK_POSITIONS, job.positions)};

    auto at{position_at(job.bounds, first)};
    for (auto index{first}; index < end; ++index, advance(at, job.bounds)) {
      if (!formation_matches(job.blocks, job.version, at))
        continue;
      ++result.found;
      if (result.matches.size() < MAX_MATCHES)
        result.matches.push_back({at.x, at.y, at.z});
    }
  }
}

} // namespace

DirectionResult run_search(const SearchBounds& bounds, const SearchFormation& formation, int version) {
  // Tops first: a top check rejects 75% of candidates, a side check 50%.
  auto blocks{formation.tops};
  blocks.insert(blocks.end(), formation.sides.begin(), formation.sides.end());
  if (blocks.empty())
    throw std::invalid_argument{"empty formation"};

  constexpr auto limit{std::numeric_limits<std::uint64_t>::max()};
  const auto nx{bounds.nx()}, ny{bounds.ny()}, nz{bounds.nz()};
  if (ny > limit / nz || nx > limit / (ny * nz))
    throw std::invalid_argument{"search volume too large"};

  Job job{.bounds = bounds, .blocks = blocks, .version = version, .positions = nx * ny * nz};
  std::vector<WorkerResult> workers(std::max(1u, std::thread::hardware_concurrency()));

  const auto started{std::chrono::steady_clock::now()};
  {
    std::vector<std::jthread> threads;
    for (auto& worker : workers)
      threads.emplace_back([&job, &worker] { search_chunks(job, worker); });
  } // joins
  const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - started};

  DirectionResult result{.kernel_seconds = elapsed.count()};
  for (const auto& worker : workers) {
    result.total += worker.found;
    const auto room{MAX_MATCHES - result.matches.size()};
    const auto take{std::min(room, worker.matches.size())};
    result.matches.insert(result.matches.end(), worker.matches.begin(), worker.matches.begin() + take);
  }
  result.truncated = result.total > MAX_MATCHES;
  std::ranges::sort(result.matches, {}, [](const MatchResult& m) { return std::tuple{m.x, m.y, m.z}; });
  return result;
}

bool shared_search_enabled(int) {
  return false;
}

SharedSearchResult run_shared_search(const SearchBounds& bounds, std::span<const SearchFormation> formations,
                                     int version) {
  SharedSearchResult result;
  for (const auto& formation : formations) {
    result.directions.push_back(run_search(bounds, formation, version));
    result.kernel_seconds += result.directions.back().kernel_seconds;
  }
  return result;
}
