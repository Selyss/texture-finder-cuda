#include "bitplanes/plan.hpp"

#include <algorithm>
#include <limits>

namespace bitplanes {

namespace {

constexpr std::int64_t INT_LOWEST{std::numeric_limits<int>::min()};
constexpr std::int64_t INT_HIGHEST{std::numeric_limits<int>::max()};

[[nodiscard]] int coset_of(const BlockInfo& block) {
  return residue(-std::int64_t{block.z});
}

[[nodiscard]] DirectionPlan plan_direction(std::span<const BlockInfo> blocks, const SearchBounds& bounds,
                                           int min_x, std::span<const int> plane_y) {
  DirectionPlan plan;
  for (int coset{0}; coset < STRIDE; ++coset) {
    for (const auto& block : blocks) {
      if (coset_of(block) == coset) {
        const auto level{std::lower_bound(plane_y.begin(), plane_y.end(), bounds.y_min + block.y)};
        plan.checks.push_back({.x_offset = block.x - min_x,
                               .y_slot = static_cast<int>(level - plane_y.begin()),
                               .z_offset = block.z,
                               .rotation = block.rotation,
                               .is_side = block.is_side});
      } else {
        plan.verify.push_back({.z_term = std::int64_t{block.z} * 116129781LL,
                               .x_term = static_cast<std::uint32_t>(block.x) * 3129871u,
                               .y_offset = block.y,
                               .rotation = block.rotation,
                               .mask = block.is_side ? 1 : 3});
      }
    }
    plan.offsets[coset + 1] = static_cast<int>(plan.checks.size());
    plan.verify_offsets[coset + 1] = static_cast<int>(plan.verify.size());
  }
  return plan;
}

} // namespace

bool eligible(std::span<const BlockInfo> blocks) {
  std::array<int, STRIDE> filter_bits{};
  for (const auto& block : blocks)
    filter_bits[coset_of(block)] += block.is_side ? 1 : 2;
  return *std::min_element(filter_bits.begin(), filter_bits.end()) >= 4;
}

std::optional<SearchPlan> plan_search(const SearchBounds& bounds,
                                      std::span<const std::vector<BlockInfo>> formations,
                                      std::size_t plane_bytes) {
  int min_x{formations.front().front().x}, max_x{min_x};
  int min_z{formations.front().front().z}, max_z{min_z};
  std::vector<int> y_offsets;
  for (const auto& blocks : formations)
    for (const auto& block : blocks) {
      min_x = std::min(min_x, block.x);
      max_x = std::max(max_x, block.x);
      min_z = std::min(min_z, block.z);
      max_z = std::max(max_z, block.z);
      y_offsets.push_back(block.y);
    }
  std::sort(y_offsets.begin(), y_offsets.end());
  y_offsets.erase(std::unique(y_offsets.begin(), y_offsets.end()), y_offsets.end());

  const auto nx{static_cast<std::int64_t>(bounds.nx())};
  const auto ny{static_cast<std::int64_t>(bounds.ny())};
  const auto nz{static_cast<std::int64_t>(bounds.nz())};
  const auto field_x{std::int64_t{bounds.x_min} + min_x};
  const auto width{nx + std::int64_t{max_x} - min_x};
  const auto halo{std::int64_t{max_z} - min_z};

  // Every coordinate a kernel forms must fit in int.
  if (nx <= 0 || ny <= 0 || nz <= 0 || width > INT_HIGHEST || ny > INT_HIGHEST || nz > INT_HIGHEST ||
      field_x < INT_LOWEST || field_x + width - 1 > INT_HIGHEST ||
      std::int64_t{bounds.z_min} + min_z < INT_LOWEST || std::int64_t{bounds.z_max} + max_z > INT_HIGHEST ||
      std::int64_t{bounds.y_min} + y_offsets.front() < INT_LOWEST ||
      std::int64_t{bounds.y_max} + y_offsets.back() > INT_HIGHEST)
    return std::nullopt;

  const std::size_t field_words{static_cast<std::size_t>(ceil_div(static_cast<int>(width), WORD_BITS)) + 1};
  const auto candidate_words{ceil_div(static_cast<int>(nx), WORD_BITS)};
  const auto segments{ceil_div(candidate_words, THREADS)};
  const auto words_budget{plane_bytes / sizeof(Planes)};
  if (static_cast<std::size_t>(ny) > words_budget / field_words)
    return std::nullopt;

  auto plane_y{compact_levels(bounds, y_offsets)};
  if (plane_y.size() > words_budget / field_words)
    return std::nullopt;

  // The field for one slab holds every STRIDE-th z row of the slab plus its halo.
  const auto words_per_row{plane_y.size() * field_words};
  const auto rows_that_fit{static_cast<std::int64_t>(words_budget / words_per_row) * STRIDE - halo};
  const auto match_blocks_per_row{static_cast<std::size_t>(ny) * segments};
  const auto grid_limit{INT_HIGHEST / static_cast<std::int64_t>(match_blocks_per_row)};
  const auto slab_rows{std::min<std::int64_t>({nz, DEFAULT_SLAB_ROWS, rows_that_fit, grid_limit})};
  if (slab_rows <= 0)
    return std::nullopt;

  SearchPlan plan{.bounds = bounds,
                  .min_x = min_x,
                  .min_z = min_z,
                  .max_z = max_z,
                  .field_x = static_cast<int>(field_x),
                  .field_width = static_cast<int>(width),
                  .field_words = static_cast<int>(field_words),
                  .candidate_words = candidate_words,
                  .plane_y = std::move(plane_y),
                  .slab_rows = static_cast<int>(slab_rows),
                  .plane_words = words_per_row * ((slab_rows + halo + STRIDE - 1) / STRIDE),
                  .match_blocks_per_row = match_blocks_per_row,
                  .directions = {}};
  for (const auto& blocks : formations)
    plan.directions.push_back(plan_direction(blocks, bounds, min_x, plan.plane_y));
  return plan;
}

Slab SearchPlan::slab(std::int64_t z_offset) const {
  const auto nz{static_cast<std::int64_t>(bounds.nz())};
  const auto rows{static_cast<int>(std::min<std::int64_t>(slab_rows, nz - z_offset))};
  auto slab_bounds{bounds};
  slab_bounds.z_min = static_cast<int>(bounds.z_min + z_offset);
  slab_bounds.z_max = slab_bounds.z_min + (rows - 1);

  // Hashed rows are the multiples of STRIDE between the slab's halo ends.
  const auto halo_start{std::int64_t{slab_bounds.z_min} + min_z};
  const auto halo_end{std::int64_t{slab_bounds.z_max} + max_z};
  const auto first_hashed_z{halo_start + residue(-halo_start)};
  const auto hashed_rows{
      first_hashed_z <= halo_end ? static_cast<int>((halo_end - first_hashed_z) / STRIDE + 1) : 0};
  const auto levels{static_cast<int>(plane_y.size())};
  return {.layout = {.bounds = slab_bounds,
                     .nx = static_cast<int>(bounds.nx()),
                     .ny = static_cast<int>(bounds.ny()),
                     .nz = rows,
                     .field_x = field_x,
                     .field_width = field_width,
                     .field_words = field_words,
                     .candidate_words = candidate_words,
                     .y_slots = levels,
                     .hashed_rows = hashed_rows,
                     .first_hashed_z = first_hashed_z},
          .field_words = plane_y.size() * field_words * hashed_rows,
          .match_blocks = match_blocks_per_row * rows};
}

} // namespace bitplanes
