#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/bitplane_search.hpp"
#include "backends/cuda/device.hpp"
#include "bitplanes/plan.hpp"
#include "diagnostic_kernels.cuh"
#include "texture/formation_io.hpp"
#include "texture/rotation.hpp"
#include "texture/texture_rules.hpp"

namespace {
using namespace bitplanes;
using namespace cuda_bitplanes;
using namespace cuda_bitplanes::diagnostic;
using cuda::DeviceArray;
constexpr int SAMPLES = 9;
using Counts = std::array<Word, STRIDE>;

constexpr std::array match_modes{MatchMode::Full, MatchMode::Filter, MatchMode::NoMemory, MatchMode::Div32,
                                 MatchMode::NoTail};
constexpr std::array mode_names{"full", "filter", "filter-nomem", "filter-div32", "filter-notail"};

struct Sample {
  float baseline, fill, match_and_verify;
  double baseline_wall, prototype_wall;
};

struct SearchResult {
  std::vector<MatchResult> matches;
  Word total = 0;
};

int checked_int(int64_t value) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::runtime_error("Experiment bounds exceed the supported integer range");
  return static_cast<int>(value);
}

size_t product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::runtime_error("Experiment size overflows size_t");
  return a * b;
}

unsigned int grid_size(size_t count) {
  if (count > std::numeric_limits<int>::max())
    throw std::runtime_error("Experiment exceeds the CUDA grid size");
  return static_cast<unsigned int>(count);
}

template <class Launch>
float time_kernel(Launch launch) {
  cuda::Timer timer;
  launch();
  cuda::check(cudaGetLastError());
  return static_cast<float>(timer.stop_seconds() * 1000.0);
}

// Large enough that the planner never refuses a field. The harness sizes its own.
constexpr size_t UNLIMITED_PLANE_BYTES = size_t{1} << 48;

SearchPlan checked_plan(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations) {
  auto plan = plan_search(bounds, formations, UNLIMITED_PLANE_BYTES);
  if (!plan)
    throw std::runtime_error("Experiment bounds are empty or exceed the supported integer range");
  return std::move(*plan);
}

// The production plan for one direction of a search. The field covers every
// formation passed in, so directions planned together can share it.
struct Experiment {
  SearchPlan plan;
  DirectionPlan direction;
  Layout layout; // the whole volume, before it is cut into slabs
  bool world_y;
  const std::vector<BlockInfo>& formation;

  Experiment(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations, size_t index,
             bool compact_y)
      : plan(checked_plan(bounds, formations))
      , direction(plan.directions[index])
      , world_y(compact_y)
      , formation(formations[index]) {
    if (!world_y) {
      // The earlier field kept one level per block y offset and repeated it for every candidate y.
      std::vector<int> y_offsets;
      for (const auto& block : formation)
        y_offsets.push_back(block.y);
      std::sort(y_offsets.begin(), y_offsets.end());
      y_offsets.erase(std::unique(y_offsets.begin(), y_offsets.end()), y_offsets.end());
      for (auto& check : direction.checks) {
        const int y_offset = plan.plane_y[check.y_slot] - bounds.y_min;
        check.y_slot = std::lower_bound(y_offsets.begin(), y_offsets.end(), y_offset) - y_offsets.begin();
      }
      plan.plane_y = y_offsets;
    }
    layout = {bounds,
              static_cast<int>(bounds.nx()),
              static_cast<int>(bounds.ny()),
              static_cast<int>(bounds.nz()),
              plan.field_x,
              plan.field_width,
              plan.field_words,
              plan.candidate_words,
              checked_int(plan.plane_y.size()),
              0,
              0};
  }

  Experiment(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks, bool compact_y)
      : Experiment(bounds, std::span{&blocks, 1}, 0, compact_y) {}

  size_t rows() const { return product(layout.ny, layout.nz); }
};

// Evaluate each origin directly, independently of GPU plane construction and extraction.
Word reference_mask(const Layout& layout, int y, int z, int word, const std::vector<BlockInfo>& blocks) {
  const int start = word * WORD_BITS;
  const int bits = std::min(WORD_BITS, layout.nx - start);
  Word mask = 0;
  for (int bit = 0; bit < bits; ++bit) {
    const int x = layout.bounds.x_min + start + bit;
    bool passes = true;
    for (const auto& block : blocks) {
      const int mod = block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM;
      if (get_texture_modern(x + block.x, y + block.y, z + block.z, mod) != block.rotation) {
        passes = false;
        break;
      }
    }
    if (passes)
      mask |= 1ull << bit;
  }
  return mask;
}

Counts reference_counts(const Experiment& experiment, unsigned int threads, const Word* masks = nullptr) {
  const auto& layout = experiment.layout;
  const size_t rows = experiment.rows();
  auto check_rows = [&](size_t first, size_t last) {
    Counts counts{};
    std::vector<BlockInfo> selected;
    selected.reserve(experiment.formation.size());
    for (size_t row = first; row < last; ++row) {
      const int z = layout.bounds.z_min + static_cast<int>(row % layout.nz);
      const int y = layout.bounds.y_min + static_cast<int>(row / layout.nz);
      const int coset = residue(z);
      selected.clear();
      for (const auto& block : experiment.formation)
        if ((z + block.z) % STRIDE == 0)
          selected.push_back(block);
      for (int word = 0; word < layout.candidate_words; ++word) {
        const Word expected = reference_mask(layout, y, z, word, selected);
        const size_t index = row * layout.candidate_words + word;
        if (masks && masks[index] != expected)
          throw std::runtime_error("GPU/CPU mask mismatch at word " + std::to_string(index));
        counts[coset] += __builtin_popcountll(expected);
      }
    }
    return counts;
  };

  const size_t workers = std::min<size_t>(threads, rows);
  std::vector<std::future<Counts>> work;
  for (size_t i = 0; i < workers; ++i)
    work.push_back(std::async(std::launch::async, check_rows, rows * i / workers, rows * (i + 1) / workers));
  Counts totals{};
  for (auto& worker : work) {
    const auto counts = worker.get();
    for (int coset = 0; coset < STRIDE; ++coset)
      totals[coset] += counts[coset];
  }
  return totals;
}

class DeviceExperiment {
public:
  DeviceExperiment(const Experiment& experiment, int slab_rows, bool capture_masks,
                   size_t capacity = MAX_MATCHES, bool precompute = false, bool own_field = true)
      : layout_(experiment.layout)
      , plan_(experiment.plan)
      , slab_rows_(std::min(slab_rows, layout_.nz))
      , world_y_(experiment.world_y)
      , precompute_(precompute)
      , field_words_per_z_(product(product(world_y_ ? 1 : layout_.ny, layout_.y_slots), layout_.field_words))
      , field_capacity_(
            product(field_words_per_z_,
                    (static_cast<int64_t>(slab_rows_) + plan_.max_z - plan_.min_z + STRIDE - 1) / STRIDE))
      , segments_(ceil_div(layout_.candidate_words, THREADS))
      , match_capacity_(grid_size(product(product(layout_.ny, slab_rows_), segments_)))
      , capacity_(capacity)
      , planes_(own_field ? field_capacity_ : 0)
      , checks_(precompute ? 0 : experiment.direction.checks.size())
      , plane_checks_(precompute ? experiment.direction.checks.size() : 0)
      , slab_checks_(precompute ? experiment.direction.checks.size() : 0)
      , experiment_(experiment)
      , offsets_(STRIDE + 1)
      , y_offsets_(experiment.plan.plane_y.size())
      , counts_(match_capacity_)
      , masks_(capture_masks ? product(experiment.rows(), layout_.candidate_words) : 0)
      , verify_(experiment.direction.verify.size())
      , verify_offsets_(STRIDE + 1)
      , matches_(capacity_)
      , total_matches_(1) {
    plan_.slab_rows = slab_rows_;
    if (!precompute_)
      checks_.upload(experiment.direction.checks);
    offsets_.upload(experiment.direction.offsets);
    y_offsets_.upload(experiment.plan.plane_y);
    verify_.upload(experiment.direction.verify);
    verify_offsets_.upload(experiment.direction.verify_offsets);
  }

  SearchResult run(Sample& sample, Counts* survivor_counts = nullptr, MatchMode mode = MatchMode::Full,
                   bool pack32 = false) {
    reset_results();
    sample.fill = 0;
    sample.match_and_verify = 0;
    std::vector<unsigned int> host_counts;
    if (survivor_counts && mode == MatchMode::NoTail)
      throw std::logic_error("No-tail mode does not produce block counts");
    if (precompute_ && mode == MatchMode::NoMemory)
      throw std::logic_error("Synthetic words are only wired into the unprecomputed filter");
    if (survivor_counts) {
      survivor_counts->fill(0);
      host_counts.resize(match_capacity_);
    }

    for (int offset = 0; offset < layout_.nz;) {
      const Layout slab = slab_at(offset);
      sample.fill += fill_slab(slab, pack32);
      sample.match_and_verify += match_slab(slab, planes_.get(), offset, mode);
      if (survivor_counts)
        tally_slab(slab, host_counts, *survivor_counts);
      offset += slab.nz;
    }
    if (mode != MatchMode::Full && mode != MatchMode::Search)
      return {};
    return results();
  }

  void reset_results() { total_matches_.zero(); }

  const Planes* field() const { return planes_.get(); }

  // The production slab at this z offset, cut to the harness's own slab height.
  Layout slab_at(int offset) const { return plan_.slab(offset).layout; }

  float fill_slab(const Layout& slab, bool pack32 = false) {
    const size_t field_count = field_words_per_z_ * slab.hashed_rows;
    const unsigned int fill_grid = grid_size(field_count / THREADS + (field_count % THREADS != 0));
    if (fill_grid)
      return time_kernel([&] {
        if (pack32)
          fill_planes<Version::modern><<<fill_grid, THREADS>>>(slab, y_offsets_.get(), field_count,
                                                               planes_.get());
        else if (world_y_)
          fill_planes_plain<true><<<fill_grid, THREADS>>>(slab, y_offsets_.get(), field_count, planes_.get());
        else
          fill_planes_plain<false><<<fill_grid, THREADS>>>(slab, y_offsets_.get(), field_count,
                                                           planes_.get());
      });
    return 0;
  }

  unsigned int blocks_in_slab(const Layout& slab) const {
    return product(product(slab.ny, slab.nz), segments_);
  }

  float match_slab(const Layout& slab, const Planes* field, int offset, MatchMode mode) {
    const unsigned int match_grid = blocks_in_slab(slab);
    Word* masks = masks_.get();
    if (masks)
      masks += static_cast<size_t>(offset) * slab.candidate_words;
    if (precompute_) {
      prepare_plane_checks(slab, experiment_.direction.checks, experiment_.direction.offsets.data(),
                           slab_checks_);
      plane_checks_.upload(slab_checks_);
    }
    return time_kernel([&] {
      if (world_y_)
        launch_match<true>(mode, slab, match_grid, field, masks);
      else
        launch_match<false>(mode, slab, match_grid, field, masks);
    });
  }

  void tally_slab(const Layout& slab, std::vector<unsigned int>& host_counts, Counts& counts) {
    const auto grid = blocks_in_slab(slab);
    host_counts = counts_.download(grid);
    for (size_t block = 0; block < grid; ++block) {
      const int z = slab.bounds.z_min + static_cast<int>((block / segments_) % slab.nz);
      counts[residue(z)] += host_counts[block];
    }
  }

  SearchResult results() {
    SearchResult result;
    result.total = total_matches_.download(1).front();
    result.matches = matches_.download(std::min<Word>(capacity_, result.total));
    std::sort(result.matches.begin(), result.matches.end(),
              [](const auto& a, const auto& b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); });
    return result;
  }

  std::vector<Word> masks() const { return masks_.download(masks_.size()); }
  size_t plane_bytes() const { return planes_.get() ? field_capacity_ * sizeof(Planes) : 0; }
  int slab_count() const { return ceil_div(layout_.nz, slab_rows_); }

private:
  template <bool WorldY>
  void launch_match(MatchMode mode, Layout slab, unsigned int grid, const Planes* field, Word* masks) {
    const auto launch = [&](auto tag) {
      if constexpr (WorldY) {
        if (precompute_) {
          match_words_probe<decltype(tag)::value, true, true><<<grid, THREADS>>>(
              slab, plane_checks_.get(), offsets_.get(), field, counts_.get(), masks, layout_.nz,
              verify_.get(), verify_offsets_.get(), matches_.get(), total_matches_.get(), capacity_);
          return;
        }
      }
      match_words_probe<decltype(tag)::value, WorldY><<<grid, THREADS>>>(
          slab, checks_.get(), offsets_.get(), field, counts_.get(), masks, layout_.nz, verify_.get(),
          verify_offsets_.get(), matches_.get(), total_matches_.get(), capacity_);
    };
    switch (mode) {
    case MatchMode::Full:
      launch(std::integral_constant<MatchMode, MatchMode::Full>{});
      break;
    case MatchMode::Filter:
      launch(std::integral_constant<MatchMode, MatchMode::Filter>{});
      break;
    case MatchMode::NoMemory:
      launch(std::integral_constant<MatchMode, MatchMode::NoMemory>{});
      break;
    case MatchMode::Div32:
      launch(std::integral_constant<MatchMode, MatchMode::Div32>{});
      break;
    case MatchMode::NoTail:
      launch(std::integral_constant<MatchMode, MatchMode::NoTail>{});
      break;
    case MatchMode::Search:
      launch(std::integral_constant<MatchMode, MatchMode::Search>{});
      break;
    }
  }

  Layout layout_;
  SearchPlan plan_;
  int slab_rows_;
  bool world_y_, precompute_;
  size_t field_words_per_z_, field_capacity_;
  int segments_;
  unsigned int match_capacity_;
  size_t capacity_;
  DeviceArray<Planes> planes_;
  DeviceArray<Check> checks_;
  DeviceArray<PlaneCheck> plane_checks_;
  std::vector<PlaneCheck> slab_checks_;
  const Experiment& experiment_;
  DeviceArray<int> offsets_, y_offsets_;
  DeviceArray<unsigned int> counts_;
  DeviceArray<Word> masks_;
  DeviceArray<VerifyCheck> verify_;
  DeviceArray<int> verify_offsets_;
  DeviceArray<MatchResult> matches_;
  DeviceArray<Word> total_matches_;
};

bool same_match(const MatchResult& a, const MatchResult& b) {
  return std::tie(a.x, a.y, a.z) == std::tie(b.x, b.y, b.z);
}

bool is_match(const MatchResult& match, const SearchBounds& bounds, const std::vector<BlockInfo>& formation) {
  if (match.x < bounds.x_min || match.x > bounds.x_max || match.y < bounds.y_min || match.y > bounds.y_max ||
      match.z < bounds.z_min || match.z > bounds.z_max)
    return false;
  return std::all_of(formation.begin(), formation.end(), [&](const BlockInfo& block) {
    const int mod = block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM;
    return get_texture_modern(match.x + block.x, match.y + block.y, match.z + block.z, mod) == block.rotation;
  });
}

void compare_results(const SearchResult& actual, const SearchResult& expected, const Experiment& experiment,
                     size_t capacity = MAX_MATCHES) {
  if (actual.total != expected.total || actual.matches.size() != std::min<Word>(actual.total, capacity))
    throw std::runtime_error("Prototype/baseline result counts differ");
  if (actual.total <= capacity) {
    if (!std::equal(actual.matches.begin(), actual.matches.end(), expected.matches.begin(),
                    expected.matches.end(), same_match))
      throw std::runtime_error("Prototype/baseline match sets differ");
    return;
  }
  if (std::adjacent_find(actual.matches.begin(), actual.matches.end(), same_match) != actual.matches.end() ||
      !std::all_of(actual.matches.begin(), actual.matches.end(), [&](const MatchResult& match) {
        return is_match(match, experiment.layout.bounds, experiment.formation);
      }))
    throw std::runtime_error("Prototype stored an invalid or duplicate match");
}

SearchResult baseline_search(const SearchBounds& bounds, const std::vector<BlockInfo>& tops,
                             const std::vector<BlockInfo>& sides, float& kernel_ms) {
  auto blocks = tops;
  blocks.insert(blocks.end(), sides.begin(), sides.end());
  auto result = run_search(bounds, blocks, Version::modern);
  kernel_ms = static_cast<float>(result.kernel_seconds * 1000.0);
  return {std::move(result.matches), result.total};
}

// The backend's own bitplane search, planned and launched as the CLI runs it.
void expect_production(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations,
                       std::span<const SearchResult> expected) {
  const auto output = cuda_bitplanes::search(bounds, formations, Version::modern);
  if (!output)
    throw std::runtime_error("Production bitplane search did not fit in memory");
  for (size_t i = 0; i < formations.size(); ++i) {
    const Experiment experiment(bounds, formations[i], true);
    compare_results({output->directions[i].matches, output->directions[i].total}, expected[i], experiment);
  }
}

void verify_small_case(bool world_y, int candidate_levels, bool sparse_levels, bool precompute = false) {
  // Two x words plus a partial word, mixed faces, unaligned and multiword offsets,
  // negative coordinates, and a z interval covering every coset twice.
  const SearchBounds bounds{-70, 60, -2, -2 + candidate_levels - 1, -7, 4};
  std::vector<BlockInfo> formation{{0, -1, -1, 0, false}, {63, 2, 0, 1, false}, {65, 0, 1, 2, false},
                                   {-1, 0, 2, 1, true},   {3, 1, 3, 0, true},   {127, 0, 4, 3, false},
                                   {2, 1, 5, 0, true},    {7, 0, 6, 2, false}};
  if (sparse_levels) {
    const std::array offsets{-5, -2, 0, 50};
    for (size_t i = 0; i < formation.size(); ++i)
      formation[i].y = offsets[i % offsets.size()];
  }
  const MatchResult known{1, bounds.y_max, bounds.z_min + 5};
  std::vector<BlockInfo> tops, sides;
  for (auto& block : formation) {
    const int mod = block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM;
    block.rotation = get_texture_modern(known.x + block.x, known.y + block.y, known.z + block.z, mod);
    (block.is_side ? sides : tops).push_back(block);
  }
  const Experiment experiment(bounds, formation, world_y);
  const size_t expected_levels = candidate_levels == 1 ? 4 : 11;
  if (world_y && sparse_levels && experiment.plan.plane_y.size() != expected_levels)
    throw std::runtime_error("Unexpected compact level count for sparse offsets");
  DeviceExperiment device(experiment, sparse_levels ? 1 : 5, true, MAX_MATCHES, precompute);
  Sample sample{};
  Counts survivors;
  const auto actual = device.run(sample, &survivors);
  if (survivors != reference_counts(experiment, 1, device.masks().data()))
    throw std::runtime_error("GPU/CPU count mismatch on the small case");
  const auto expected = baseline_search(bounds, tops, sides, sample.baseline);
  compare_results(actual, expected, experiment);
  expect_production(bounds, std::span{&formation, 1}, std::span{&expected, 1});
  if (std::none_of(actual.matches.begin(), actual.matches.end(),
                   [&](const auto& match) { return same_match(match, known); }))
    throw std::runtime_error("Prototype missed the known synthetic match");
  for (const auto mode : {MatchMode::Filter, MatchMode::Div32, MatchMode::NoTail}) {
    device.run(sample, nullptr, mode);
    reference_counts(experiment, 1, device.masks().data());
  }
  std::puts("PASS small GPU/CPU masks and exact matches across slab boundaries");
  if (sparse_levels)
    return;

  const std::vector<BlockInfo> single{{0, 0, 0, 0, false}};
  const Experiment dense(bounds, single, world_y);
  if (world_y && dense.plan.plane_y.size() != static_cast<size_t>(candidate_levels))
    throw std::runtime_error("Unexpected compact level count for one offset");
  constexpr size_t small_capacity = 3;
  DeviceExperiment limited(dense, 1, false, small_capacity, precompute);
  const auto dense_expected = baseline_search(bounds, single, {}, sample.baseline);
  if (dense_expected.total <= small_capacity)
    throw std::runtime_error("Dense synthetic case did not exceed result capacity");
  compare_results(limited.run(sample), dense_expected, dense, small_capacity);
  std::puts("PASS bounded result storage and true match count across slabs");
}

void run_diagnostics(const Experiment& experiment, int slab_rows, const Counts& expected, bool all_modes) {
  const auto selected = [&](MatchMode mode) {
    return all_modes || mode == MatchMode::Filter || mode == MatchMode::NoMemory;
  };
  DeviceExperiment device(experiment, slab_rows, false);
  for (MatchMode mode : match_modes) {
    if (!selected(mode))
      continue;
    Sample warmup{};
    Counts counts{};
    device.run(warmup, mode == MatchMode::NoTail ? nullptr : &counts, mode);
    if (mode == MatchMode::NoMemory) {
      Word real_total = 0, synthetic_total = 0;
      for (int coset = 0; coset < STRIDE; ++coset) {
        real_total += expected[coset];
        synthetic_total += counts[coset];
      }
      std::printf("DIAG_SURVIVORS real=%llu synthetic=%llu\n", real_total, synthetic_total);
    } else if (mode != MatchMode::NoTail && counts != expected)
      throw std::runtime_error("Diagnostic survivor counts differ from CPU reference");
  }
  for (int sample = 0; sample < SAMPLES; ++sample) {
    for (size_t offset = 0; offset < match_modes.size(); ++offset) {
      const size_t index = (sample + offset) % match_modes.size();
      const MatchMode mode = match_modes[index];
      if (!selected(mode))
        continue;
      Sample timing{};
      device.run(timing, nullptr, mode);
      std::printf("DIAG %d %s fill_ms=%.6f match_ms=%.6f\n", sample + 1, mode_names[index], timing.fill,
                  timing.match_and_verify);
    }
  }
}

void compare_plane_offsets(const Experiment& experiment, int slab_rows, const SearchResult& reference) {
  if (!experiment.world_y)
    throw std::runtime_error("Offset comparison requires compact world-y planes");
  verify_small_case(true, 4, false, true);
  verify_small_case(true, 3, true, true);
  verify_small_case(true, 1, true, true);

  Sample warmup{};
  Counts original_counts{}, offset_counts{};
  for (bool precompute : {false, true}) {
    DeviceExperiment device(experiment, slab_rows, false, MAX_MATCHES, precompute);
    auto& counts = precompute ? offset_counts : original_counts;
    compare_results(device.run(warmup, &counts), reference, experiment);
    device.run(warmup, nullptr, MatchMode::NoTail);
  }
  if (original_counts != offset_counts)
    throw std::runtime_error("Precomputed offsets changed survivor counts");
  for (int coset = 0; coset < STRIDE; ++coset)
    std::printf("COSET %d %llu\n", coset, offset_counts[coset]);

  using Clock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  for (int i = 0; i < SAMPLES; ++i) {
    std::array<Sample, 2> search{}, filter{};
    for (int j = 0; j < 2; ++j) {
      const bool precompute = (i + j) % 2 != 0;
      auto& sample = search[precompute];
      SearchResult result;
      const auto start = Clock::now();
      {
        DeviceExperiment device(experiment, slab_rows, false, MAX_MATCHES, precompute);
        result = device.run(sample, nullptr, MatchMode::Search);
      }
      sample.prototype_wall = Milliseconds(Clock::now() - start).count();
      compare_results(result, reference, experiment);

      DeviceExperiment device(experiment, slab_rows, false, MAX_MATCHES, precompute);
      device.run(filter[precompute], nullptr, MatchMode::NoTail);
    }
    std::printf("OFFSET_PAIR %d %.6f %.6f %.6f %.6f %.6f %.6f\n", i + 1, filter[0].match_and_verify,
                filter[1].match_and_verify, search[0].match_and_verify, search[1].match_and_verify,
                search[0].prototype_wall, search[1].prototype_wall);
  }
  std::puts("PASS precomputed offsets preserve masks, survivor counts and exact matches");
}

struct DirectionTiming {
  float fill = 0, match = 0;
  double wall = 0;
};

void compare_shared_field(const SearchBounds& bounds, const std::vector<BlockInfo>& formation, int slab_rows,
                          bool check_masks) {
  constexpr int directions = 4;
  std::array<std::vector<BlockInfo>, directions> turned;
  for (int dir = 0; dir < directions; ++dir)
    turned[dir] = rotated(formation, dir);
  std::array<std::optional<Experiment>, directions> independent, shared;
  std::array<SearchResult, directions> expected;
  std::array<Counts, directions> counts;
  std::array<std::vector<Word>, directions> masks;
  std::array<size_t, directions> individual_bytes{};
  for (int dir = 0; dir < directions; ++dir) {
    const auto& direction = turned[dir];
    independent[dir].emplace(bounds, direction, true);
    shared[dir].emplace(bounds, turned, dir, true);
    const auto side = std::find_if(direction.begin(), direction.end(),
                                   [](const BlockInfo& block) { return block.is_side; });
    float baseline_ms = 0;
    expected[dir] = baseline_search(bounds, {direction.begin(), side}, {side, direction.end()}, baseline_ms);
    Sample sample{};
    DeviceExperiment device(*independent[dir], slab_rows, check_masks, MAX_MATCHES, true);
    compare_results(device.run(sample, &counts[dir]), expected[dir], *independent[dir]);
    individual_bytes[dir] = device.plane_bytes();
    if (check_masks) {
      masks[dir] = device.masks();
      reference_counts(*independent[dir], 1, masks[dir].data());
    }
  }

  expect_production(bounds, turned, expected);

  const auto run = [&](bool reuse, bool validate) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    DirectionTiming time;
    if (reuse) {
      std::array<std::unique_ptr<DeviceExperiment>, directions> devices;
      for (int dir = 0; dir < directions; ++dir) {
        devices[dir] = std::make_unique<DeviceExperiment>(*shared[dir], slab_rows, validate && check_masks,
                                                          MAX_MATCHES, true, dir == 0);
        devices[dir]->reset_results();
      }
      auto& owner = *devices.front();
      const auto mode = validate ? MatchMode::Full : MatchMode::Search;
      std::array<Counts, directions> actual_counts{};
      std::vector<unsigned int> scratch;
      for (int offset = 0; offset < shared[0]->layout.nz;) {
        const auto slab = owner.slab_at(offset);
        time.fill += owner.fill_slab(slab);
        for (int dir = 0; dir < directions; ++dir) {
          time.match += devices[dir]->match_slab(slab, owner.field(), offset, mode);
          if (validate)
            devices[dir]->tally_slab(slab, scratch, actual_counts[dir]);
        }
        offset += slab.nz;
      }
      for (int dir = 0; dir < directions; ++dir) {
        compare_results(devices[dir]->results(), expected[dir], *independent[dir]);
        if (validate && actual_counts[dir] != counts[dir])
          throw std::runtime_error("Shared field changed survivor counts");
        if (validate && check_masks) {
          if (devices[dir]->masks() != masks[dir])
            throw std::runtime_error("Shared field changed a survivor mask");
        }
      }
      if (validate && !check_masks) {
        std::printf("SHARED_BYTES %zu %zu %zu %zu %zu\n", owner.plane_bytes(), individual_bytes[0],
                    individual_bytes[1], individual_bytes[2], individual_bytes[3]);
        for (int dir = 0; dir < directions; ++dir) {
          std::printf("DIRECTION %d matches=%llu survivors", dir, expected[dir].total);
          for (Word count : actual_counts[dir])
            std::printf(" %llu", count);
          std::putchar('\n');
        }
      }
    } else {
      for (int dir = 0; dir < directions; ++dir) {
        Sample sample{};
        DeviceExperiment device(*independent[dir], slab_rows, false, MAX_MATCHES, true);
        compare_results(device.run(sample, nullptr, MatchMode::Search), expected[dir], *independent[dir]);
        time.fill += sample.fill;
        time.match += sample.match_and_verify;
      }
    }
    time.wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return time;
  };
  run(true, true);
  if (check_masks)
    return;
  run(false, false);
  run(true, false);
  for (int i = 0; i < SAMPLES; ++i) {
    std::array<DirectionTiming, 2> pair;
    for (int j = 0; j < 2; ++j) {
      const bool reuse = (i + j) % 2 != 0;
      pair[reuse] = run(reuse, false);
    }
    std::printf("SHARED_PAIR %d %.6f %.6f %.6f %.6f %.6f %.6f\n", i + 1, pair[0].fill, pair[1].fill,
                pair[0].match, pair[1].match, pair[0].wall, pair[1].wall);
  }
  std::puts("PASS shared field preserves counts and exact results in all directions");
}

struct FusedCheck {
  int shift, level, rotation;
};

struct FusedLayout {
  SearchBounds bounds;
  int min_x, min_y, y_span, x_span, width, words, hashed_rows;
  int64_t first_z;
  int selected_z[STRIDE], filter_offsets[STRIDE + 1], verify_offsets[STRIDE + 1];
};

template <MatchMode Mode>
__device__ void fused_candidates(Word mask, FusedLayout layout, int coset, int x, int y, int z, int word,
                                 const VerifyCheck* verify, Word* masks, Word& survivors, MatchResult* output,
                                 Word* total) {
  if constexpr (Mode == MatchMode::NoTail)
    asm volatile("" ::"l"(mask));
  else {
    if constexpr (Mode == MatchMode::Full) {
      survivors += __popcll(mask);
      if (masks) {
        const size_t row =
            static_cast<size_t>(y - layout.bounds.y_min) * (layout.bounds.z_max - layout.bounds.z_min + 1) +
            (z - layout.bounds.z_min);
        masks[row * layout.words + word] = mask;
      }
    }
    const int64_t z_term = static_cast<int64_t>(z) * 116129781LL;
    for (; mask; mask &= mask - 1) {
      const int candidate_x = x + __ffsll(mask) - 1;
      const uint32_t x_term = static_cast<uint32_t>(candidate_x) * 3129871u;
      bool passes = true;
      for (int i = layout.verify_offsets[coset]; i < layout.verify_offsets[coset + 1]; ++i) {
        const auto check = verify[i];
        if (modern_from_coord_random(
                coord_random_from_parts(x_term + check.x_term, z_term + check.z_term, y + check.y_offset),
                4) != check.rotation) {
          passes = false;
          break;
        }
      }
      if (passes) {
        const Word index = atomicAdd(total, 1ull);
        if (index < MAX_MATCHES)
          output[index] = {candidate_x, y, z};
      }
    }
  }
}

template <MatchMode Mode>
__global__ void fused_search(FusedLayout layout, const FusedCheck* checks, const VerifyCheck* verify,
                             Word* masks, Word* counts, MatchResult* output, Word* total) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = index / layout.words;
  const int word = index % layout.words;
  const int x = layout.bounds.x_min + word * layout.width;
  const int bits = min(layout.width, layout.bounds.x_max - x + 1);
  const Word valid = low_mask(bits);
  Word survivors[STRIDE]{};
  if (row < layout.hashed_rows) {
    const int64_t hashed_z = layout.first_z + static_cast<int64_t>(row) * STRIDE;
    const int64_t z_term = hashed_z * 116129781LL;
    const uint32_t first_x = static_cast<uint32_t>(x + layout.min_x) * 3129871u;
    Planes previous{}, older{};
    for (int y = layout.bounds.y_min - layout.y_span; y <= layout.bounds.y_max; ++y) {
      Planes current{};
      uint32_t x_term = first_x;
      for (int bit = 0; bit < bits + layout.x_span; ++bit, x_term += 3129871u) {
        const unsigned int rotation = modern_from_coord_random(
            coord_random_from_parts(x_term, z_term, y + layout.min_y + layout.y_span), 4);
        current.low |= static_cast<Word>(rotation & 1u) << bit;
        current.high |= static_cast<Word>(rotation >> 1) << bit;
      }
      if (y >= layout.bounds.y_min) {
        for (int coset = 0; coset < STRIDE; ++coset) {
          const int first = layout.filter_offsets[coset];
          const int end = layout.filter_offsets[coset + 1];
          const int64_t z = hashed_z - layout.selected_z[coset];
          if (first == end || z < layout.bounds.z_min || z > layout.bounds.z_max)
            continue;
          Word mask = valid;
          for (int i = first; i < end && mask; ++i) {
            const auto check = checks[i];
            const auto plane = check.level == 0 ? current : (check.level == 1 ? previous : older);
            const Word low = plane.low >> check.shift;
            const Word high = plane.high >> check.shift;
            mask &= (check.rotation & 1) ? low : ~low;
            mask &= (check.rotation & 2) ? high : ~high;
          }
          fused_candidates<Mode>(mask, layout, coset, x, y, static_cast<int>(z), word, verify, masks,
                                 survivors[coset], output, total);
        }
      }
      older = previous;
      previous = current;
    }
  } else {
    const int z = layout.bounds.z_min + (row - layout.hashed_rows);
    if (z > layout.bounds.z_max)
      return;
    const int coset = residue(z);
    if (layout.filter_offsets[coset] != layout.filter_offsets[coset + 1])
      return;
    for (int y = layout.bounds.y_min; y <= layout.bounds.y_max; ++y)
      fused_candidates<Mode>(valid, layout, coset, x, y, z, word, verify, masks, survivors[coset], output,
                             total);
  }
  if constexpr (Mode == MatchMode::Full)
    for (int coset = 0; coset < STRIDE; ++coset)
      if (survivors[coset])
        atomicAdd(counts + coset, survivors[coset]);
}

struct FusedExperiment {
  FusedLayout layout{};
  std::vector<FusedCheck> filters;
  std::vector<VerifyCheck> verify;
  std::array<std::vector<BlockInfo>, STRIDE> selected;
  int grid_rows;

  FusedExperiment(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks) {
    layout.bounds = bounds;
    const auto [min_x, max_x] = std::minmax_element(blocks.begin(), blocks.end(),
                                                    [](const auto& a, const auto& b) { return a.x < b.x; });
    const auto [min_y, max_y] = std::minmax_element(blocks.begin(), blocks.end(),
                                                    [](const auto& a, const auto& b) { return a.y < b.y; });
    layout.min_x = min_x->x;
    layout.min_y = min_y->y;
    layout.x_span = max_x->x - min_x->x;
    layout.y_span = max_y->y - min_y->y;
    if (STRIDE != 5 || layout.x_span >= WORD_BITS || layout.y_span > 2 ||
        std::any_of(blocks.begin(), blocks.end(), [](const auto& b) { return b.is_side; }))
      throw std::runtime_error("Fused experiment requires stride 5, top faces, x span < 64 and y span <= 2");
    layout.width = WORD_BITS - layout.x_span;
    layout.words = ceil_div(checked_int(static_cast<int64_t>(bounds.x_max) - bounds.x_min + 1), layout.width);
    std::map<int, int> by_z;
    for (const auto& b : blocks)
      ++by_z[b.z];
    std::array<int, STRIDE> best{};
    for (const auto& [z, count] : by_z) {
      const int coset = residue(-static_cast<int64_t>(z));
      if (count > best[coset]) {
        best[coset] = count;
        layout.selected_z[coset] = z;
      }
    }
    for (int coset = 0; coset < STRIDE; ++coset) {
      for (const auto& b : blocks) {
        if (best[coset] && b.z == layout.selected_z[coset]) {
          filters.push_back({b.x - layout.min_x, max_y->y - b.y, b.rotation});
          selected[coset].push_back(b);
        } else
          verify.push_back({static_cast<int64_t>(b.z) * 116129781LL, static_cast<uint32_t>(b.x) * 3129871u,
                            b.y, b.rotation, 3});
      }
      layout.filter_offsets[coset + 1] = static_cast<int>(filters.size());
      layout.verify_offsets[coset + 1] = static_cast<int>(verify.size());
    }
    const int64_t start = static_cast<int64_t>(bounds.z_min) + by_z.begin()->first;
    const int64_t end = static_cast<int64_t>(bounds.z_max) + by_z.rbegin()->first;
    layout.first_z = start + residue(-start);
    layout.hashed_rows = layout.first_z <= end ? checked_int((end - layout.first_z) / STRIDE + 1) : 0;
    grid_rows = layout.hashed_rows;
    if (std::find(best.begin(), best.end(), 0) != best.end())
      grid_rows = checked_int(static_cast<int64_t>(grid_rows) + bounds.z_max - bounds.z_min + 1);
  }

  size_t mask_words() const {
    const auto& b = layout.bounds;
    return product(product(layout.words, b.z_max - b.z_min + 1), b.y_max - b.y_min + 1);
  }
};

class DeviceFused {
public:
  DeviceFused(const FusedExperiment& experiment, bool capture)
      : experiment_(experiment)
      , filters_(experiment.filters.size())
      , verify_(experiment.verify.size())
      , masks_(capture ? experiment.mask_words() : 0)
      , counts_(STRIDE)
      , output_(MAX_MATCHES)
      , total_(1) {
    filters_.upload(experiment.filters);
    verify_.upload(experiment.verify);
  }

  template <MatchMode Mode>
  float run() {
    total_.zero();
    counts_.zero();
    const auto& layout = experiment_.layout;
    const size_t work = product(layout.words, experiment_.grid_rows);
    if (!work)
      return 0;
    return time_kernel([&] {
      fused_search<Mode><<<grid_size((work + THREADS - 1) / THREADS), THREADS>>>(
          layout, filters_.get(), verify_.get(), masks_.get(), counts_.get(), output_.get(), total_.get());
    });
  }

  SearchResult results() const {
    SearchResult result;
    result.total = total_.download(1).front();
    result.matches = output_.download(std::min<Word>(result.total, MAX_MATCHES));
    std::sort(result.matches.begin(), result.matches.end(),
              [](const auto& a, const auto& b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); });
    return result;
  }

  Counts counts() const {
    Counts result;
    const auto values = counts_.download(STRIDE);
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  }

  void compare_masks() const {
    const auto masks = masks_.download(experiment_.mask_words());
    const auto& layout = experiment_.layout;
    const auto& b = layout.bounds;
    size_t index = 0;
    for (int y = b.y_min; y <= b.y_max; ++y)
      for (int z = b.z_min; z <= b.z_max; ++z)
        for (int word = 0; word < layout.words; ++word) {
          Word expected = 0;
          const int x = b.x_min + word * layout.width;
          const int bits = std::min(layout.width, b.x_max - x + 1);
          for (int bit = 0; bit < bits; ++bit) {
            const auto& blocks = experiment_.selected[residue(z)];
            const bool passes = std::all_of(blocks.begin(), blocks.end(), [&](const auto& block) {
              return get_texture_modern(x + bit + block.x, y + block.y, z + block.z, 4) == block.rotation;
            });
            if (passes)
              expected |= 1ull << bit;
          }
          if (masks[index++] != expected)
            throw std::runtime_error("Fused GPU mask differs from CPU oracle");
        }
  }

private:
  const FusedExperiment& experiment_;
  DeviceArray<FusedCheck> filters_;
  DeviceArray<VerifyCheck> verify_;
  DeviceArray<Word> masks_, counts_;
  DeviceArray<MatchResult> output_;
  DeviceArray<Word> total_;
};

void verify_fused(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks,
                  const Counts* expected_counts, bool capture) {
  const Experiment planes(bounds, blocks, true);
  const FusedExperiment experiment(bounds, blocks);
  DeviceFused fused(experiment, capture);
  fused.run<MatchMode::Full>();
  if (capture)
    fused.compare_masks();
  if (expected_counts && fused.counts() != *expected_counts)
    throw std::runtime_error("Unexpected fused survivor counts");
  float ms = 0;
  compare_results(fused.results(), baseline_search(bounds, blocks, {}, ms), planes);
}

Word fused_cpu_coset_zero(const FusedExperiment& experiment, unsigned int threads) {
  // Count independently of GPU packing. Only coset zero changes for the reference.
  const auto& layout = experiment.layout;
  const auto& bounds = layout.bounds;
  const int64_t first = static_cast<int64_t>(bounds.z_min) + residue(-static_cast<int64_t>(bounds.z_min));
  const size_t rows = first <= bounds.z_max ? (bounds.z_max - first) / STRIDE + 1 : 0;
  const auto& checks = experiment.selected[0];
  std::vector<std::future<Word>> workers;
  for (unsigned int i = 0; i < threads; ++i)
    workers.push_back(std::async(std::launch::async, [&, i] {
      Word count = 0;
      for (size_t row = rows * i / threads; row < rows * (i + 1) / threads; ++row) {
        const int z = static_cast<int>(first + static_cast<int64_t>(row) * STRIDE);
        for (int y = bounds.y_min; y <= bounds.y_max; ++y)
          for (int x = bounds.x_min; x <= bounds.x_max; ++x)
            if (std::all_of(checks.begin(), checks.end(), [&](const auto& b) {
                  return get_texture_modern(x + b.x, y + b.y, z + b.z, 4) == b.rotation;
                }))
              ++count;
      }
      return count;
    }));
  Word total = 0;
  for (auto& worker : workers)
    total += worker.get();
  return total;
}

void time_fused(const char* label, const SearchBounds& bounds, const std::vector<BlockInfo>& blocks,
                int slab_rows, unsigned int threads, bool reference) {
  const FusedExperiment experiment(bounds, blocks);
  const Experiment planes(bounds, blocks, true);
  DeviceFused fused(experiment, false);
  DeviceExperiment current(planes, slab_rows, false, MAX_MATCHES, true);
  float standard_ms = 0;
  const auto expected = baseline_search(bounds, blocks, {}, standard_ms);
  if (reference && expected.total != 1)
    throw std::runtime_error("Full reference must return exactly one match");
  fused.run<MatchMode::Full>();
  const auto counts = fused.counts();
  compare_results(fused.results(), expected, planes);
  if (reference) {
    const Counts cpu{fused_cpu_coset_zero(experiment, threads), 24417246, 24412805, 24406584, 390623920};
    if (counts != cpu)
      throw std::runtime_error("Full fused counts differ from CPU oracle");
  }
  Word survivors = 0;
  for (Word count : counts)
    survivors += count;
  const auto& layout = experiment.layout;
  const auto candidates = static_cast<Word>(layout.bounds.x_max - layout.bounds.x_min + 1) *
                          (layout.bounds.y_max - layout.bounds.y_min + 1) *
                          (layout.bounds.z_max - layout.bounds.z_min + 1);
  std::printf("FUSED_COUNTS %s matches=%llu survivors=%llu candidates=%llu cosets", label, expected.total,
              survivors, candidates);
  for (Word count : counts)
    std::printf(" %llu", count);
  std::putchar('\n');
  Sample warm{};
  fused.run<MatchMode::NoTail>();
  fused.run<MatchMode::Search>();
  current.run(warm, nullptr, MatchMode::NoTail);
  compare_results(current.run(warm, nullptr, MatchMode::Search), expected, planes);
  for (int i = 0; i < SAMPLES; ++i) {
    Sample filter{}, search{};
    float fused_filter = 0, fused_total = 0;
    for (int j = 0; j < 2; ++j) {
      if ((i + j) % 2) {
        fused_filter = fused.run<MatchMode::NoTail>();
        fused_total = fused.run<MatchMode::Search>();
        compare_results(fused.results(), expected, planes);
      } else {
        current.run(filter, nullptr, MatchMode::NoTail);
        compare_results(current.run(search, nullptr, MatchMode::Search), expected, planes);
      }
    }
    std::printf("FUSED_PAIR %s %d %.6f %.6f %.6f %.6f %.6f\n", label, i + 1, filter.fill,
                filter.match_and_verify, search.fill + search.match_and_verify, fused_filter, fused_total);
  }
  std::printf("PASS fused %s exact matches in all timed samples\n", label);
}

void run_fused_experiment(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks, int slab_rows,
                          unsigned int threads) {
  const SearchBounds tiny{-78, 67, -3, 3, -7, -4};
  std::vector<BlockInfo> synthetic{
      {-4, -2, -6, 0, false}, {1, 0, -6, 0, false}, {6, -1, 2, 0, false}, {0, 0, 7, 0, false}};
  for (auto& b : synthetic)
    b.rotation = get_texture_modern(-11 + b.x, -1 + b.y, -6 + b.z, 4);
  verify_fused(tiny, synthetic, nullptr, true);
  auto one_row = tiny;
  one_row.z_min = one_row.z_max;
  verify_fused(one_row, synthetic, nullptr, true);
  const SearchBounds slice{bounds.x_min, bounds.x_min + 19999, bounds.y_min,
                           bounds.y_max, bounds.z_min,         bounds.z_min + 119};
  const Counts expected{46920, 11776, 11649, 11527, 188452};
  verify_fused(slice, blocks, &expected, true);
  std::puts(
      "PASS fused synthetic and reference slice masks and exact matches; slice counts 46920 11776 11649 11527 188452");

  time_fused("reference", bounds, blocks, slab_rows, threads, true);
  std::vector<BlockInfo> line;
  for (int z = 0; z < 12; ++z)
    line.push_back({0, 0, z, z % 4, false});
  SearchBounds line_bounds = bounds;
  line_bounds.y_max = line_bounds.y_min;
  time_fused("line", line_bounds, line, slab_rows, threads, false);
}

void compare_packing(const SearchBounds& bounds, const std::vector<BlockInfo>& formation, int slab_rows) {
  const Experiment experiment(bounds, formation, true);
  DeviceExperiment original(experiment, slab_rows, false);
  DeviceExperiment packed(experiment, slab_rows, false);
  const auto check = [&](int offset) {
    const auto slab = original.slab_at(offset);
    const size_t count = product(product(slab.y_slots, slab.hashed_rows), slab.field_words);
    original.fill_slab(slab);
    packed.fill_slab(slab, true);
    // Compare in chunks to keep host memory small even on a full slab.
    constexpr size_t chunk = 1 << 18;
    std::vector<Planes> a(chunk), b(chunk);
    for (size_t i = 0; i < count; i += chunk) {
      const auto n = std::min(chunk, count - i);
      cuda::check(cudaMemcpy(a.data(), original.field() + i, n * sizeof(Planes), cudaMemcpyDeviceToHost));
      cuda::check(cudaMemcpy(b.data(), packed.field() + i, n * sizeof(Planes), cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < n; ++j)
        if (a[j].low != b[j].low || a[j].high != b[j].high)
          throw std::runtime_error("32-bit packing changed a plane word");
    }
    return count;
  };
  std::printf("PACK_CHECK words=%zu\n", check(0));
  if (experiment.layout.nz <= slab_rows)
    return;

  const auto time_fill = [&](bool pack32) {
    float ms = 0;
    auto& device = pack32 ? packed : original;
    for (int offset = 0; offset < experiment.layout.nz;) {
      const auto slab = device.slab_at(offset);
      ms += device.fill_slab(slab, pack32);
      offset += slab.nz;
    }
    return ms;
  };
  time_fill(false);
  time_fill(true);
  for (int i = 0; i < SAMPLES; ++i) {
    std::array<float, 2> times{};
    for (int j = 0; j < 2; ++j) {
      const bool pack32 = (i + j) % 2 != 0;
      times[pack32] = time_fill(pack32);
    }
    std::printf("PACK_PAIR %d %.6f %.6f\n", i + 1, times[0], times[1]);
  }
}

void compare_packing_search(const SearchBounds& bounds, const std::vector<BlockInfo>& formation,
                            int slab_rows) {
  const Experiment experiment(bounds, formation, true);
  const auto side =
      std::find_if(formation.begin(), formation.end(), [](const BlockInfo& block) { return block.is_side; });
  float standard_ms = 0;
  const auto expected =
      baseline_search(bounds, {formation.begin(), side}, {side, formation.end()}, standard_ms);
  if (expected.total != 1)
    throw std::runtime_error("Expected one match on the packing reference workload");

  const auto run = [&](bool pack32) {
    Sample sample{};
    SearchResult result;
    const auto start = std::chrono::steady_clock::now();
    {
      DeviceExperiment device(experiment, slab_rows, false, MAX_MATCHES, true);
      result = device.run(sample, nullptr, MatchMode::Search, pack32);
    }
    sample.prototype_wall =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    compare_results(result, expected, experiment);
    return sample;
  };
  run(false);
  run(true);
  for (int i = 0; i < SAMPLES; ++i) {
    std::array<Sample, 2> pair{};
    for (int j = 0; j < 2; ++j) {
      const bool pack32 = (i + j) % 2 != 0;
      pair[pack32] = run(pack32);
    }
    std::printf("PACK_SEARCH %d %.6f %.6f %.6f %.6f %.6f %.6f\n", i + 1, pair[0].fill, pair[1].fill,
                pair[0].fill + pair[0].match_and_verify, pair[1].fill + pair[1].match_and_verify,
                pair[0].prototype_wall, pair[1].prototype_wall);
  }
  std::puts("PASS packing search exact standard-kernel match in every run");
}

// The standard kernel's result for blocks listed tops first.
SearchResult standard_result(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks) {
  const auto side =
      std::find_if(blocks.begin(), blocks.end(), [](const BlockInfo& block) { return block.is_side; });
  float ms = 0;
  return baseline_search(bounds, {blocks.begin(), side}, {side, blocks.end()}, ms);
}

int parse_int(std::string_view value) {
  int result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::runtime_error("Expected an integer argument");
  return result;
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 8 || argc > 10) {
    std::fprintf(stderr,
                 "Usage: %s FIXTURE X_MIN X_MAX Y_MIN Y_MAX Z_MIN Z_MAX "
                 "[CPU_THREADS [SLAB_ROWS]]\n",
                 argv[0]);
    return 1;
  }

  try {
    // run_search is the baseline every experiment is compared with, so it must
    // stay on the standard kernel. The production bitplanes are called directly.
    setenv("TF_CUDA_BITPLANES", "0", 1);
    const SearchBounds bounds{parse_int(argv[2]), parse_int(argv[3]), parse_int(argv[4]),
                              parse_int(argv[5]), parse_int(argv[6]), parse_int(argv[7])};
    const int threads = argc >= 9 ? parse_int(argv[8]) : 1;
    const int slab_rows = argc == 10 ? parse_int(argv[9]) : DEFAULT_SLAB_ROWS;
    if (threads < 1 || slab_rows < 1)
      throw std::runtime_error("CPU_THREADS and SLAB_ROWS must be positive");

    auto formation = parse_formation_file(argv[1]);
    const auto sides_begin = std::stable_partition(formation.begin(), formation.end(),
                                                   [](const BlockInfo& block) { return !block.is_side; });
    const std::vector<BlockInfo> tops(formation.begin(), sides_begin);
    const std::vector<BlockInfo> sides(sides_begin, formation.end());
    if (std::getenv("TF_FLAT_COMPARE_PACKING")) {
      const std::vector<BlockInfo> synthetic{{-4, -2, -3, 0, false}, {0, 0, 3, 0, false}};
      for (int width : {1, 31, 32, 33, 63, 64, 65, 127})
        compare_packing({-131, -131 + width - 1, -3, 2, -12, 8}, synthetic, 21);
      compare_packing(bounds, formation, slab_rows);
      return 0;
    }
    if (std::getenv("TF_FLAT_COMPARE_PACKING_SEARCH")) {
      compare_packing_search(bounds, formation, slab_rows);
      return 0;
    }
    if (std::getenv("TF_FLAT_FUSED")) {
      run_fused_experiment(bounds, formation, slab_rows, threads);
      return 0;
    }
    if (std::getenv("TF_FLAT_SHARED_FIELD")) {
      const SearchBounds small{-70, 60, -2, 1, -7, 4};
      std::vector<BlockInfo> synthetic{
          {-77, -1, -3, 0, false}, {18, 2, 67, 0, false}, {2, 0, 4, 0, false}, {66, 1, -1, 0, true}};
      for (auto& block : synthetic)
        block.rotation =
            get_texture_modern(1 + block.x, block.y, -2 + block.z, block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM);
      compare_shared_field(small, synthetic, 1, true);
      for (int dir = 1; dir < 4; ++dir) {
        synthetic = rotated(std::move(synthetic), 1);
        compare_shared_field(small, synthetic, 5, true);
      }
      std::puts("PASS shared masks match independent GPU fields and CPU on synthetic slabs");
      compare_shared_field(bounds, formation, slab_rows, false);
      return 0;
    }
    const bool world_y = std::getenv("TF_FLAT_DUPLICATE_Y") == nullptr;
    const Experiment experiment(bounds, formation, world_y);
    if (std::getenv("TF_FLAT_COMPARE_OFFSETS")) {
      float baseline_ms = 0;
      const auto reference = baseline_search(bounds, tops, sides, baseline_ms);
      compare_plane_offsets(experiment, slab_rows, reference);
      return 0;
    }
    verify_small_case(world_y, 4, false);
    if (world_y) {
      verify_small_case(world_y, 3, true);
      verify_small_case(world_y, 1, true);
    }
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const auto baseline = [&](Sample& sample) {
      const auto start = Clock::now();
      auto result = baseline_search(bounds, tops, sides, sample.baseline);
      sample.baseline_wall = Milliseconds(Clock::now() - start).count();
      return result;
    };
    const auto prototype = [&](Sample& sample) {
      const auto start = Clock::now();
      SearchResult result;
      {
        DeviceExperiment device(experiment, slab_rows, false);
        result = device.run(sample);
      }
      sample.prototype_wall = Milliseconds(Clock::now() - start).count();
      return result;
    };
    Sample warmup{};
    const auto reference = baseline(warmup);
    expect_production(bounds, std::span{&formation, 1}, std::span{&reference, 1});
    std::puts("PASS production bitplane search matches the standard kernel");
    Counts counts;
    {
      DeviceExperiment device(experiment, slab_rows, false);
      std::printf("SEARCH candidates=%zu plane_bytes=%zu slabs=%d\n",
                  product(experiment.rows(), experiment.layout.nx), device.plane_bytes(),
                  device.slab_count());
      compare_results(device.run(warmup, &counts), reference, experiment);
    }

    std::array<Sample, SAMPLES> samples;
    for (int i = 0; i < SAMPLES; ++i) {
      auto& sample = samples[i];
      SearchResult actual, expected;
      if (i % 2 == 0) {
        expected = baseline(sample);
        actual = prototype(sample);
      } else {
        actual = prototype(sample);
        expected = baseline(sample);
      }
      compare_results(actual, reference, experiment);
      compare_results(expected, reference, experiment);
    }
    const Counts expected = reference_counts(experiment, threads);
    if (counts != expected)
      throw std::runtime_error("GPU/CPU survivor counts differ on the search volume");
    for (int coset = 0; coset < STRIDE; ++coset)
      std::printf("COSET %d %llu\n", coset, expected[coset]);
    std::printf("PASS full GPU/CPU survivor counts and baseline matches (total=%llu)\n", reference.total);
    if (const char* diagnostics = std::getenv("TF_FLAT_DIAGNOSTICS"))
      run_diagnostics(experiment, slab_rows, expected, std::string_view(diagnostics) == "all");
    for (int i = 0; i < SAMPLES; ++i)
      std::printf("SAMPLE %d baseline_ms=%.6f fill_ms=%.6f match_verify_ms=%.6f "
                  "baseline_wall_ms=%.6f prototype_wall_ms=%.6f\n",
                  i + 1, samples[i].baseline, samples[i].fill, samples[i].match_and_verify,
                  samples[i].baseline_wall, samples[i].prototype_wall);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
