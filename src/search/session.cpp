#include "search/session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include "search/checkpoint.hpp"
#include "search/progress.hpp"
#include "texture/rotation.hpp"

namespace {

// Larger volumes are searched in z slices, so the session can report
// progress, print matches as they arrive, and checkpoint between slices.
constexpr std::uint64_t SLICE_THRESHOLD{1'000'000'000};
constexpr std::uint64_t FIRST_SLICE_POSITIONS{200'000'000};
constexpr double TARGET_SLICE_SECONDS{2.0};

constexpr std::array<std::string_view, 4> DIRECTION_NAMES{"North", "West", "South", "East"};
constexpr std::array<std::string_view, NUM_VERSIONS> VERSION_NAMES{"1.21.2+", "1.13 - 1.21.1", "<=1.12.2",
                                                                   "Sodium 1.0-4.1 (MC 1.16 - 1.18.2)",
                                                                   "Sodium 4.2-4.8 (MC 1.19 - 1.19.3)"};

void print_header(const SearchRequest& request) {
  std::cout << "Backend: " << search_backend_name() << "\n"
            << "Version: " << VERSION_NAMES[request.version] << "\n";
  if (request.directions.size() == 1)
    std::cout << "Facing: " << DIRECTION_NAMES[request.directions.front()] << "\n";
  else
    std::cout << "Facing: trying all directions\n";
  std::cout.flush();
}

// Top faces first, each group in file order: a top check rejects 75% of
// candidates and a side check 50%, so backends test the tops first.
[[nodiscard]] std::vector<BlockInfo> tops_first(std::vector<BlockInfo> blocks) {
  const auto first_side{std::ranges::stable_partition(blocks, std::not_fn(&BlockInfo::is_side)).begin()};
  const auto tops{static_cast<std::size_t>(first_side - blocks.begin())};
  if (tops > MAX_FORMATION_BLOCKS || blocks.size() - tops > MAX_FORMATION_BLOCKS)
    throw std::runtime_error("formation has too many blocks (max " + std::to_string(MAX_FORMATION_BLOCKS) +
                             " per face type)");
  return blocks;
}

[[nodiscard]] std::vector<BlockInfo> rotated(std::vector<BlockInfo> blocks, int quarter_turns) {
  for (int turn{0}; turn < quarter_turns; ++turn)
    for (auto& block : blocks)
      rotate_formation(block);
  return blocks;
}

// Positions in one z layer. Throws if the whole volume overflows 64 bits.
[[nodiscard]] std::uint64_t checked_layer_size(const SearchBounds& bounds) {
  constexpr auto limit{std::numeric_limits<std::uint64_t>::max()};
  const auto nx{bounds.nx()}, ny{bounds.ny()}, nz{bounds.nz()};
  if (nx > limit / ny || nx * ny > limit / nz)
    throw std::runtime_error("search volume too large");
  return nx * ny;
}

// Each top block divides coincidental matches by 4 and each side block by 2.
// Says how much constraint is missing when the formation cannot be unique.
void warn_if_weak(std::span<const BlockInfo> formation, const SearchBounds& bounds) {
  const auto volume{static_cast<double>(bounds.nx()) * static_cast<double>(bounds.ny()) *
                    static_cast<double>(bounds.nz())};
  const auto sides{std::ranges::count_if(formation, &BlockInfo::is_side)};
  const auto have_bits{2.0 * static_cast<double>(formation.size()) - static_cast<double>(sides)};
  const auto need_bits{std::log2(volume) + 6.64}; // at most ~0.01 expected coincidences
  if (need_bits <= have_bits)
    return;
  const auto more_tops{static_cast<int>(std::ceil((need_bits - have_bits) / 2.0))};
  std::cout << "Warning: formation is weak for this volume; ~" << volume * std::pow(2.0, -have_bits)
            << " coincidental matches expected. Add ~" << more_tops << " more top-face blocks (or "
            << 2 * more_tops << " side blocks) for a unique result." << std::endl;
}

void print_match(const MatchResult& match, int direction, bool name_direction) {
  std::cout << "Match found at [" << match.x << ", " << match.y << ", " << match.z << "]";
  if (name_direction)
    std::cout << " facing " << DIRECTION_NAMES[direction];
  std::cout << "\n";
}

[[nodiscard]] Checkpoint open_checkpoint(const SearchRequest& request, bool slice_first) {
  const auto& bounds{request.bounds};
  Checkpoint checkpoint;
  checkpoint.slice_first = slice_first;
  checkpoint.next_z = bounds.z_min;
  if (request.checkpoint_path.empty())
    return checkpoint;

  checkpoint.path = request.checkpoint_path;
  checkpoint.signature = checkpoint_signature(request);
  if (!checkpoint.load())
    return checkpoint;

  // The file decides the slice order from here on, so check it fits this request.
  const auto z_in_range{checkpoint.next_z >= bounds.z_min &&
                        checkpoint.next_z <= std::int64_t{bounds.z_max} + 1};
  if (checkpoint.slice_first &&
      (request.directions.size() == 1 || checkpoint.direction_index != 0 || !z_in_range))
    throw std::runtime_error("invalid slice-first checkpoint: " + checkpoint.path);
  std::cout << "Resuming from checkpoint: direction index " << checkpoint.direction_index
            << ", z = " << checkpoint.next_z << std::endl;
  return checkpoint;
}

// TF_MAX_SLICES=N stops after N slices and leaves the checkpoint behind, so
// tests can exercise resume. Negative: no limit.
[[nodiscard]] std::int64_t slice_limit_from_environment() {
  const auto* text{std::getenv("TF_MAX_SLICES")};
  return text ? std::atoll(text) : -1;
}

[[nodiscard]] SearchBounds z_slice(SearchBounds bounds, std::int64_t first, std::int64_t last) {
  bounds.z_min = static_cast<int>(first);
  bounds.z_max = static_cast<int>(last);
  return bounds;
}

class Session {
public:
  Session(const SearchRequest& request, const std::vector<BlockInfo>& formation, std::uint64_t layer_size)
      : bounds_{request.bounds}
      , version_{request.version}
      , directions_{request.directions}
      , formation_{formation}
      , layer_size_{layer_size}
      , name_directions_{directions_.size() > 1}
      , checkpoint_{open_checkpoint(request, name_directions_ && shared_search_enabled(version_))}
      , slice_limit_{slice_limit_from_environment()} {
    const auto volume{layer_size_ * bounds_.nz()};
    sliced_ = checkpoint_.active() || volume > SLICE_THRESHOLD;
    if (!sliced_)
      return;
    const auto layers_done{static_cast<std::uint64_t>(checkpoint_.next_z - bounds_.z_min)};
    const auto directions_per_layer{checkpoint_.slice_first ? directions_.size() : 1};
    progress_.start(volume * directions_.size(),
                    checkpoint_.direction_index * volume + layers_done * layer_size_ * directions_per_layer);
  }

  [[nodiscard]] SearchOutcome run() {
    const auto finished{checkpoint_.slice_first ? search_slice_first() : search_direction_first()};
    if (!finished)
      return {SessionStatus::stopped};
    progress_.finish();
    print_summary();
    checkpoint_.remove();
    return {SessionStatus::finished, totals().kernel_seconds};
  }

private:
  struct Found {
    MatchResult match;
    int direction;
  };

  [[nodiscard]] Totals& totals() noexcept { return checkpoint_.totals; }

  // Every direction is searched within each z slice. False if stopped early.
  [[nodiscard]] bool search_slice_first() {
    std::vector<std::vector<BlockInfo>> formations;
    for (const int direction : directions_)
      formations.push_back(rotated(formation_, direction));
    const auto positions_per_layer{layer_size_ * directions_.size()};

    for (auto z{checkpoint_.next_z}; z <= bounds_.z_max;) {
      const auto last{sliced_ ? slice_end(z, positions_per_layer) : std::int64_t{bounds_.z_max}};
      const auto batch{run_shared_search(z_slice(bounds_, z, last), formations, version_)};
      for (std::size_t i{0}; i < batch.directions.size(); ++i)
        accept(batch.directions[i], directions_[i]);

      const auto positions{static_cast<std::uint64_t>(last - z + 1) * positions_per_layer};
      z = last + 1;
      if (!sliced_)
        totals().kernel_seconds += batch.kernel_seconds;
      else if (!finish_slice(positions, batch.kernel_seconds, 0, z, z <= bounds_.z_max))
        return false;
    }
    return true;
  }

  // Each direction is searched to the end before the next. False if stopped early.
  [[nodiscard]] bool search_direction_first() {
    for (auto index{checkpoint_.direction_index}; index < directions_.size(); ++index) {
      const int direction{directions_[index]};
      const auto formation{rotated(formation_, direction)};

      if (!sliced_) {
        const auto result{run_search(bounds_, formation, version_)};
        totals().kernel_seconds += result.kernel_seconds;
        accept(result, direction);
        continue;
      }

      // next_z is mid-direction only straight after a resume; otherwise it is z_min.
      for (auto z{checkpoint_.next_z}; z <= bounds_.z_max;) {
        const auto last{slice_end(z, layer_size_)};
        const auto result{run_search(z_slice(bounds_, z, last), formation, version_)};
        accept(result, direction);

        const auto positions{static_cast<std::uint64_t>(last - z + 1) * layer_size_};
        z = last + 1;
        const auto work_remains{z <= bounds_.z_max || index + 1 < directions_.size()};
        if (!finish_slice(positions, result.kernel_seconds, index, z, work_remains))
          return false;
      }
      checkpoint_.direction_index = index + 1;
      checkpoint_.next_z = bounds_.z_min;
      checkpoint_.save();
    }
    return true;
  }

  // Last z of the slice starting at `z`, sized from the measured rate so a
  // slice takes about TARGET_SLICE_SECONDS.
  [[nodiscard]] std::int64_t slice_end(std::int64_t z, std::uint64_t positions_per_layer) const {
    const auto target{rate_ > 0 ? static_cast<std::uint64_t>(rate_ * TARGET_SLICE_SECONDS)
                                : FIRST_SLICE_POSITIONS};
    const auto layers{std::max<std::uint64_t>(1, target / positions_per_layer)};
    const auto layers_left{static_cast<std::uint64_t>(bounds_.z_max - z)};
    return layers <= layers_left ? z + static_cast<std::int64_t>(layers) - 1 : bounds_.z_max;
  }

  // A sliced search prints matches as they arrive. An unsliced one is short,
  // so it collects them and prints one sorted list at the end.
  void accept(const DirectionResult& result, int direction) {
    totals().truncated |= result.truncated;
    totals().matches += result.total;
    totals().listed += result.matches.size();
    if (!sliced_) {
      for (const auto& match : result.matches)
        collected_.push_back({match, direction});
    } else if (!result.matches.empty()) {
      progress_.clear_line();
      for (const auto& match : result.matches)
        print_match(match, direction, name_directions_);
      std::cout.flush();
    }
  }

  // Records one finished slice. False if the slice limit says to stop here.
  [[nodiscard]] bool finish_slice(std::uint64_t positions, double seconds, std::size_t direction_index,
                                  std::int64_t next_z, bool work_remains) {
    totals().kernel_seconds += seconds;
    if (seconds > 0) {
      const auto rate{static_cast<double>(positions) / seconds};
      rate_ = rate_ > 0 ? 0.5 * rate_ + 0.5 * rate : rate;
    }
    progress_.advance(positions);
    checkpoint_.direction_index = direction_index;
    checkpoint_.next_z = next_z;
    checkpoint_.save();

    ++slices_done_;
    if (slice_limit_ < 0 || slices_done_ < slice_limit_ || !work_remains)
      return true;
    progress_.finish();
    std::cout << "Stopping after " << slices_done_ << " slice(s) (TF_MAX_SLICES)";
    if (checkpoint_.active())
      std::cout << "; state saved to " << checkpoint_.path;
    std::cout << std::endl;
    return false;
  }

  void print_summary() {
    std::ranges::sort(collected_, {}, [](const Found& found) {
      return std::tuple{found.match.x, found.match.y, found.match.z, found.direction};
    });
    for (const auto& found : collected_)
      print_match(found.match, found.direction, name_directions_);

    const auto& [matches, listed, kernel_seconds, truncated]{totals()};
    if (matches == 0)
      std::cout << "No matches found." << std::endl;
    else
      std::cout << matches << (matches == 1 ? " match" : " matches") << std::endl;
    if (truncated)
      std::cout << "Warning: only the first " << listed << " of " << matches
                << " matches are listed; shrink the search area." << std::endl;
  }

  SearchBounds bounds_;
  int version_;
  std::span<const int> directions_;
  const std::vector<BlockInfo>& formation_;
  std::uint64_t layer_size_;
  bool name_directions_;
  Checkpoint checkpoint_;
  std::int64_t slice_limit_;
  std::int64_t slices_done_{0};
  bool sliced_{false};
  double rate_{0.0}; // positions per second, smoothed over slices
  Progress progress_;
  std::vector<Found> collected_;
};

} // namespace

SearchOutcome run_search_session(const SearchRequest& request) {
  print_header(request);
  const auto formation{tops_first(request.formation)};
  const auto layer_size{checked_layer_size(request.bounds)};
  warn_if_weak(formation, request.bounds);
  Session session{request, formation, layer_size};
  return session.run();
}
