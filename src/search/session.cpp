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
#include <string>
#include <string_view>
#include <tuple>

#include "search/checkpoint.hpp"
#include "search/progress.hpp"
#include "texture/rotation.hpp"
#include "texture/version_names.hpp"

namespace {

// Larger volumes are searched in z slices, so the session can report
// progress, print matches as they arrive, and checkpoint between slices.
constexpr std::uint64_t SLICE_THRESHOLD{1'000'000'000};
constexpr std::uint64_t FIRST_SLICE_POSITIONS{200'000'000};
constexpr double TARGET_SLICE_SECONDS{2.0};

constexpr std::array<std::string_view, 4> DIRECTION_NAMES{"North", "West", "South", "East"};

// Notes that are not results: warnings, and where a checkpoint stands. They
// are part of the text report, and go to stderr when stdout is for results only.
[[nodiscard]] std::ostream& notes(OutputMode output) {
  return output == OutputMode::text ? std::cout : std::cerr;
}

void print_header(const SearchRequest& request) {
  std::cout << "Backend: " << search_backend_name() << "\n"
            << "Version: " << version_name(request.version) << "\n";
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
void warn_if_weak(std::span<const BlockInfo> formation, const SearchRequest& request) {
  const auto& bounds{request.bounds};
  const auto volume{static_cast<double>(bounds.nx()) * static_cast<double>(bounds.ny()) *
                    static_cast<double>(bounds.nz()) * static_cast<double>(request.directions.size())};
  const auto sides{std::ranges::count_if(formation, &BlockInfo::is_side)};
  const auto have_bits{2.0 * static_cast<double>(formation.size()) - static_cast<double>(sides)};
  const auto need_bits{std::log2(volume) + 6.64}; // at most ~0.01 expected coincidences
  if (need_bits <= have_bits)
    return;
  const auto more_tops{static_cast<int>(std::ceil((need_bits - have_bits) / 2.0))};
  notes(request.output) << "Warning: formation is weak for this volume; ~"
                        << volume * std::pow(2.0, -have_bits) << " coincidental matches expected. Add ~"
                        << more_tops << " more top-face blocks (or " << 2 * more_tops
                        << " side blocks) for a unique result." << std::endl;
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
  notes(request.output) << "Resuming from checkpoint: direction index " << checkpoint.direction_index
                        << ", z = " << checkpoint.next_z << std::endl;
  return checkpoint;
}

// TF_MAX_SLICES=N stops after N slices and leaves the checkpoint behind, so
// tests can exercise resume. Negative: no limit.
[[nodiscard]] std::int64_t slice_limit_from_environment() {
  const auto* text{std::getenv("TF_MAX_SLICES")};
  return text ? std::atoll(text) : -1;
}

// "40 seconds", "12 minutes", "3 hours 20 minutes".
[[nodiscard]] std::string rough_duration(double seconds) {
  const auto plural{[](long long count, std::string_view unit) {
    auto text{std::to_string(count)};
    text += ' ';
    text += unit;
    if (count != 1)
      text += 's';
    return text;
  }};
  const auto whole{static_cast<long long>(std::llround(seconds))};
  if (whole < 90)
    return plural(whole, "second");
  const auto minutes{(whole + 30) / 60};
  if (minutes < 90)
    return plural(minutes, "minute");
  auto text{plural(minutes / 60, "hour")};
  if (minutes % 60) {
    text += ' ';
    text += plural(minutes % 60, "minute");
  }
  return text;
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
      , output_{request.output}
      , name_directions_{directions_.size() > 1}
      , checkpoint_{open_checkpoint(request, name_directions_ && shared_search_enabled(version_))}
      , slice_limit_{slice_limit_from_environment()} {
    const auto volume{layer_size_ * bounds_.nz()};
    sliced_ = checkpoint_.active() || volume > SLICE_THRESHOLD;
    if (!sliced_)
      return;
    const auto layers_done{static_cast<std::uint64_t>(checkpoint_.next_z - bounds_.z_min)};
    const auto directions_per_layer{checkpoint_.slice_first ? directions_.size() : 1};
    const auto total{volume * directions_.size()};
    const auto done{checkpoint_.direction_index * volume + layers_done * layer_size_ * directions_per_layer};
    positions_left_ = total - done;
    if (output_ != OutputMode::quiet)
      progress_.start(total, done);
  }

  [[nodiscard]] SearchOutcome run() {
    const auto finished{checkpoint_.slice_first ? search_slice_first() : search_direction_first()};
    if (finished) {
      progress_.finish();
      print_summary();
      checkpoint_.remove();
    } else {
      sort_collected();
    }
    return {.status = finished ? SessionStatus::finished : SessionStatus::stopped,
            .kernel_seconds = totals().kernel_seconds,
            .search = search_,
            .total_matches = totals().matches,
            .truncated = totals().truncated,
            .matches = std::move(collected_)};
  }

private:
  using Found = FoundMatch;

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
      report_search(batch.directions);
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
        report_search({&result, 1});
        totals().kernel_seconds += result.kernel_seconds;
        accept(result, direction);
        continue;
      }

      // next_z is mid-direction only straight after a resume; otherwise it is z_min.
      for (auto z{checkpoint_.next_z}; z <= bounds_.z_max;) {
        const auto last{slice_end(z, layer_size_)};
        const auto result{run_search(z_slice(bounds_, z, last), formation, version_)};
        report_search({&result, 1});
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

  // Says which search the backend used, the first time and when it changes.
  void report_search(std::span<const DirectionResult> results) {
    const auto with_bitplanes{std::ranges::count_if(results, &DirectionResult::bitplanes)};
    const auto total{static_cast<std::ptrdiff_t>(results.size())};
    auto search{with_bitplanes == 0       ? std::string{"plain"}
                : with_bitplanes == total ? std::string{"bitplanes"}
                                          : "bitplanes for " + std::to_string(with_bitplanes) + " of " +
                                                std::to_string(total) + " directions"};
    if (search == search_)
      return;
    search_ = std::move(search);
    if (output_ != OutputMode::text)
      return;
    progress_.clear_line();
    std::cout << "Search: " << search_ << std::endl;
  }

  // A sliced search prints matches as they arrive. An unsliced one is short,
  // so it collects them and prints one sorted list at the end. JSON output
  // always collects.
  void accept(const DirectionResult& result, int direction) {
    totals().truncated |= result.truncated;
    totals().matches += result.total;
    totals().listed += result.matches.size();
    if (!sliced_ || output_ == OutputMode::json) {
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
      last_rate_ = rate;
    }
    progress_.advance(positions);
    positions_left_ -= std::min(positions, positions_left_);
    checkpoint_.direction_index = direction_index;
    checkpoint_.next_z = next_z;
    checkpoint_.save();

    ++slices_done_;
    if (slices_done_ == 2)
      print_estimate();
    if (slice_limit_ < 0 || slices_done_ < slice_limit_ || !work_remains)
      return true;
    progress_.finish();
    auto& out{notes(output_)};
    out << "Stopping after " << slices_done_ << " slice(s) (TF_MAX_SLICES)";
    if (checkpoint_.active())
      out << "; state saved to " << checkpoint_.path;
    out << std::endl;
    return false;
  }

  // The first slice is short and includes the backend's warm-up, so the rate
  // of the second is the first one worth quoting. A search long enough to
  // wait for says roughly how long; the progress line refines it from there.
  void print_estimate() {
    constexpr double WORTH_SAYING_SECONDS{5.0};
    if (output_ != OutputMode::text || last_rate_ <= 0)
      return;
    const auto seconds{static_cast<double>(positions_left_) / last_rate_};
    if (seconds < WORTH_SAYING_SECONDS)
      return;
    progress_.clear_line();
    std::cout << "Estimated time: about " << rough_duration(seconds) << std::endl;
  }

  void sort_collected() {
    std::ranges::sort(collected_, {}, [](const Found& found) {
      return std::tuple{found.match.x, found.match.y, found.match.z, found.direction};
    });
  }

  void print_summary() {
    sort_collected();
    const auto& [matches, listed, kernel_seconds, truncated]{totals()};
    if (output_ != OutputMode::json) {
      for (const auto& found : collected_)
        print_match(found.match, found.direction, name_directions_);
      std::cout.flush();
    }
    if (output_ == OutputMode::text) {
      if (matches == 0)
        std::cout << "No matches found." << std::endl;
      else
        std::cout << matches << (matches == 1 ? " match" : " matches") << std::endl;
    }
    if (truncated)
      notes(output_) << "Warning: only the first " << listed << " of " << matches
                     << " matches are listed; shrink the search area." << std::endl;
  }

  SearchBounds bounds_;
  Version version_;
  std::span<const int> directions_;
  const std::vector<BlockInfo>& formation_;
  std::uint64_t layer_size_;
  OutputMode output_;
  bool name_directions_;
  Checkpoint checkpoint_;
  std::int64_t slice_limit_;
  std::int64_t slices_done_{0};
  bool sliced_{false};
  double rate_{0.0};                // positions per second, smoothed over slices
  double last_rate_{0.0};           // of the latest slice alone
  std::uint64_t positions_left_{0}; // of a sliced search
  std::string search_;              // what report_search last said
  Progress progress_;
  std::vector<Found> collected_;
};

} // namespace

std::string_view direction_name(int direction) {
  return DIRECTION_NAMES[static_cast<std::size_t>(direction)];
}

SearchOutcome run_search_session(const SearchRequest& request) {
  if (request.output == OutputMode::text)
    print_header(request);
  const auto formation{tops_first(request.formation)};
  const auto layer_size{checked_layer_size(request.bounds)};
  warn_if_weak(formation, request);
  Session session{request, formation, layer_size};
  return session.run();
}
