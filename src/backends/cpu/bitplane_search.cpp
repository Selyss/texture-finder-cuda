#include "backends/cpu/bitplane_search.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "backends/cpu/worker_matches.hpp"
#include "bitplanes/match.hpp"
#include "bitplanes/plan.hpp"

namespace cpu_bitplanes {

namespace {

using namespace bitplanes;

// A tile is the volume one thread searches at a time: small enough that its
// field stays in the core's cache, large enough that the halo of extra rows
// and words around it is a small share of the hashing.
constexpr std::int64_t TILE_X{4096};
constexpr std::int64_t TILE_Y{32};
constexpr std::int64_t TILE_Z{64};

// A thread's field never grows past this. A tile whose field would is cut
// into slabs of fewer z rows by the planner.
constexpr std::size_t TILE_PLANE_BYTES{std::size_t{64} << 20};

// Planning the whole volume with no memory limit checks only its coordinates.
constexpr std::size_t UNLIMITED_PLANE_BYTES{std::size_t{1} << 48};

[[nodiscard]] constexpr std::uint64_t tiles(std::uint64_t positions, std::int64_t tile) noexcept {
  return (positions + static_cast<std::uint64_t>(tile) - 1) / static_cast<std::uint64_t>(tile);
}

struct Job {
  SearchBounds bounds;
  std::span<const std::vector<BlockInfo>> formations;
  Version version;
  std::uint64_t tiles_x, tiles_y, tiles_z;
  std::atomic<std::uint64_t> next_tile{0};
  std::atomic<bool> unplanned{false};

  [[nodiscard]] std::uint64_t tile_count() const noexcept { return tiles_x * tiles_y * tiles_z; }

  // Tiles are numbered x fastest, then z, then y.
  [[nodiscard]] SearchBounds tile(std::uint64_t index) const noexcept {
    const auto span{[](int min, int max, std::uint64_t tile_index, std::int64_t size) {
      const auto first{min + static_cast<std::int64_t>(tile_index) * size};
      return std::pair{static_cast<int>(first), static_cast<int>(std::min<std::int64_t>(first + size - 1, max))};
    }};
    const auto [x_min, x_max]{span(bounds.x_min, bounds.x_max, index % tiles_x, TILE_X)};
    const auto [z_min, z_max]{span(bounds.z_min, bounds.z_max, index / tiles_x % tiles_z, TILE_Z)};
    const auto [y_min, y_max]{span(bounds.y_min, bounds.y_max, index / tiles_x / tiles_z, TILE_Y)};
    return {x_min, x_max, y_min, y_max, z_min, z_max};
  }
};

class Worker {
public:
  explicit Worker(Job& job)
      : job_{job}
      , found_(job.formations.size()) {}

  void run() {
    for (;;) {
      const auto index{job_.next_tile.fetch_add(1)};
      if (index >= job_.tile_count())
        return;
      search(job_.tile(index));
    }
  }

  [[nodiscard]] const WorkerMatches& found(std::size_t direction) const { return found_[direction]; }

private:
  void search(const SearchBounds& tile) {
    const auto plan{plan_search(tile, job_.formations, TILE_PLANE_BYTES)};
    if (!plan) {
      job_.unplanned = true;
      return;
    }
    if (field_.size() < plan->plane_words)
      field_.resize(plan->plane_words);

    const auto nz{static_cast<std::int64_t>(tile.nz())};
    for (std::int64_t z_offset{0}; z_offset < nz;) {
      const auto layout{plan->slab(z_offset).layout};
      fill(layout, plan->plane_y);
      for (std::size_t i{0}; i < plan->directions.size(); ++i)
        match(layout, plan->directions[i], found_[i]);
      z_offset += layout.nz;
    }
  }

  // Hashes every field word of the slab, in the order the filter indexes them.
  void fill(const Layout& layout, std::span<const int> plane_y) {
    auto* word_out{field_.data()};
    for (const int y : plane_y) {
      for (int z_index{0}; z_index < layout.hashed_rows; ++z_index) {
        const auto z_term{(layout.first_hashed_z + std::int64_t{z_index} * STRIDE) * 116129781LL};
        for (int word{0}; word < layout.field_words; ++word) {
          const auto x_offset{std::int64_t{word} * WORD_BITS};
          const auto bits{static_cast<int>(std::min<std::int64_t>(layout.field_width - x_offset, WORD_BITS))};
          *word_out++ = hash_word(job_.version, static_cast<std::uint32_t>(layout.field_x + x_offset) * 3129871u,
                                  z_term, y, bits);
        }
      }
    }
  }

  // Filters each candidate word against the field, then checks the survivors.
  void match(const Layout& layout, const DirectionPlan& direction, WorkerMatches& found) {
    checks_.resize(direction.checks.size());
    prepare_plane_checks(layout, direction.checks, direction.offsets.data(), checks_);

    for (int y_row{0}; y_row < layout.ny; ++y_row) {
      const int y{layout.bounds.y_min + y_row};
      for (int z_row{0}; z_row < layout.nz; ++z_row) {
        const int z{layout.bounds.z_min + z_row};
        const int coset{residue(z)};
        const auto z_term{std::int64_t{z} * 116129781LL};
        for (int word{0}; word < layout.candidate_words; ++word) {
          auto survivors{filter_word(layout, checks_.data(), direction.offsets.data(), field_.data(),
                                     static_cast<std::size_t>(y_row), static_cast<std::size_t>(z_row), word)};
          for (; survivors; survivors &= survivors - 1) {
            const int x{layout.bounds.x_min + word * WORD_BITS + std::countr_zero(survivors)};
            if (survivor_matches(job_.version, direction.verify.data(), direction.verify_offsets[coset],
                                 direction.verify_offsets[coset + 1], static_cast<std::uint32_t>(x) * 3129871u,
                                 z_term, y))
              found.add({x, y, z});
          }
        }
      }
    }
  }

  Job& job_;
  std::vector<WorkerMatches> found_; // one per direction
  std::vector<Planes> field_;
  std::vector<PlaneCheck> checks_;
};

} // namespace

bool enabled() {
  const auto* setting{std::getenv("TF_CPU_BITPLANES")};
  return !setting || std::string_view{setting} != "0";
}

bool trace_enabled() {
  return std::getenv("TF_CPU_BITPLANES_TRACE") != nullptr;
}

std::optional<SearchOutput> search(const SearchBounds& bounds,
                                   std::span<const std::vector<BlockInfo>> formations, Version version) {
  Job job{.bounds = bounds,
          .formations = formations,
          .version = version,
          .tiles_x = tiles(bounds.nx(), TILE_X),
          .tiles_y = tiles(bounds.ny(), TILE_Y),
          .tiles_z = tiles(bounds.nz(), TILE_Z)};

  // Every tile can be planned if the volume's coordinates are in range and a
  // full tile's field fits. The first tile is a full one unless all are smaller.
  if (!plan_search(bounds, formations, UNLIMITED_PLANE_BYTES) ||
      !plan_search(job.tile(0), formations, TILE_PLANE_BYTES))
    return std::nullopt;

  std::vector<Worker> workers;
  const auto thread_count{std::max(1u, std::thread::hardware_concurrency())};
  workers.reserve(thread_count);
  for (unsigned int i{0}; i < thread_count; ++i)
    workers.emplace_back(job);

  const auto started{std::chrono::steady_clock::now()};
  {
    std::vector<std::jthread> threads;
    for (auto& worker : workers)
      threads.emplace_back([&worker] { worker.run(); });
  } // joins
  const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - started};
  if (job.unplanned)
    throw std::logic_error{"a bitplane tile could not be planned"};

  SearchOutput output{.directions = {}, .seconds = elapsed.count()};
  for (std::size_t i{0}; i < formations.size(); ++i)
    output.directions.push_back(
        collect_matches(workers, [i](const Worker& worker) -> const WorkerMatches& { return worker.found(i); }));
  return output;
}

} // namespace cpu_bitplanes
