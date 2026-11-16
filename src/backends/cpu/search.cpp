// Portable multithreaded CPU backend. The plain scan in this file is the
// readable reference for every other search path: it tests every position
// against every block with the shared texture rules, and threads take chunks
// of the linearised volume from an atomic cursor. Formations that filter well
// are searched with bitplanes instead (bitplane_search.hpp), which must return
// the same matches.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "backends/cpu/bitplane_search.hpp"
#include "backends/cpu/worker_matches.hpp"
#include "bitplanes/shared_search.hpp"
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
  Version version;
  std::uint64_t positions;
  std::atomic<std::uint64_t> cursor{0};
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

[[nodiscard]] bool formation_matches(std::span<const BlockInfo> blocks, Version version, Position origin) {
  return std::ranges::all_of(blocks, [&](const BlockInfo& block) {
    const int mod{block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM};
    return get_texture_for_version(version, origin.x + block.x, origin.y + block.y, origin.z + block.z,
                                   mod) == block.rotation;
  });
}

void search_chunks(Job& job, WorkerMatches& found) {
  for (;;) {
    const auto first{job.cursor.fetch_add(CHUNK_POSITIONS)};
    if (first >= job.positions)
      return;
    const auto end{std::min(first + CHUNK_POSITIONS, job.positions)};

    auto at{position_at(job.bounds, first)};
    for (auto index{first}; index < end; ++index, advance(at, job.bounds)) {
      if (formation_matches(job.blocks, job.version, at))
        found.add({at.x, at.y, at.z});
    }
  }
}

[[nodiscard]] DirectionResult plain_scan(const SearchBounds& bounds, std::span<const BlockInfo> blocks,
                                         Version version, std::uint64_t positions) {
  Job job{.bounds = bounds, .blocks = blocks, .version = version, .positions = positions};
  std::vector<WorkerMatches> workers(std::max(1u, std::thread::hardware_concurrency()));

  const auto started{std::chrono::steady_clock::now()};
  {
    std::vector<std::jthread> threads;
    for (auto& worker : workers)
      threads.emplace_back([&job, &worker] { search_chunks(job, worker); });
  } // joins
  const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - started};

  auto result{
      collect_matches(workers, [](const WorkerMatches& worker) -> const WorkerMatches& { return worker; })};
  result.kernel_seconds = elapsed.count();
  return result;
}

void trace(const char* message) {
  if (cpu_bitplanes::trace_enabled())
    std::fprintf(stderr, "CPU search: %s\n", message);
}

} // namespace

DirectionResult run_search(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks,
                           Version version) {
  if (blocks.empty())
    throw std::invalid_argument{"empty formation"};

  constexpr auto limit{std::numeric_limits<std::uint64_t>::max()};
  const auto nx{bounds.nx()}, ny{bounds.ny()}, nz{bounds.nz()};
  if (ny > limit / nz || nx > limit / (ny * nz))
    throw std::invalid_argument{"search volume too large"};

  if (cpu_bitplanes::enabled() && bitplanes::eligible(blocks)) {
    if (auto output{cpu_bitplanes::search(bounds, std::span{&blocks, 1}, version)}) {
      trace("bitplanes");
      auto& result{output->directions.front()};
      result.kernel_seconds = output->seconds;
      return std::move(result);
    }
  }
  trace("plain scan");
  return plain_scan(bounds, blocks, version, nx * ny * nz);
}

bool shared_search_enabled(Version) {
  return cpu_bitplanes::enabled();
}

SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                     std::span<const std::vector<BlockInfo>> directions, Version version) {
  return bitplanes::search_sharing_field(
      bounds, directions, version,
      [&](std::span<const std::vector<BlockInfo>> shared) -> std::optional<SharedSearchResult> {
        auto output{cpu_bitplanes::search(bounds, shared, version)};
        if (!output)
          return std::nullopt;
        if (cpu_bitplanes::trace_enabled())
          std::fprintf(stderr, "CPU search: bitplanes shared (%zu directions)\n", shared.size());
        return SharedSearchResult{std::move(output->directions), output->seconds};
      });
}
