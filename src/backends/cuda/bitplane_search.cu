#include "backends/cuda/bitplane_search.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include "backends/cuda/bitplanes.cuh"
#include "backends/cuda/device.hpp"
#include "backends/cuda/devices.hpp"
#include "backends/cuda/match_buffer.hpp"
#include "bitplanes/plan.hpp"
#include "texture/version_dispatch.hpp"

namespace cuda_bitplanes {

using namespace bitplanes;

namespace {

constexpr std::size_t DEFAULT_PLANE_BYTES{std::size_t{1} << 30};
constexpr std::size_t METADATA_BYTES{std::size_t{16} << 20};

// TF_CUDA_BITPLANES_BYTES lowers the field budget. A value that does not
// parse gives no budget, which forces the standard kernel.
[[nodiscard]] std::size_t requested_plane_bytes() {
  const auto* setting{std::getenv("TF_CUDA_BITPLANES_BYTES")};
  if (!setting)
    return DEFAULT_PLANE_BYTES;
  const std::string_view text{setting};
  std::size_t bytes{0};
  const auto [end, error]{std::from_chars(text.data(), text.data() + text.size(), bytes)};
  return error == std::errc{} && end == text.data() + text.size() ? std::min(bytes, DEFAULT_PLANE_BYTES) : 0;
}

// What the device can spare for the field after the match buffers.
[[nodiscard]] std::size_t available_plane_bytes(std::size_t directions) {
  std::size_t free_bytes{0}, total_bytes{0};
  cuda::check(cudaMemGetInfo(&free_bytes, &total_bytes));
  const auto reserved{directions * MAX_MATCHES * sizeof(MatchResult) + METADATA_BYTES};
  return std::min(requested_plane_bytes(), free_bytes > reserved ? free_bytes - reserved : 0);
}

// One direction's device buffers. The plane checks are rewritten per slab.
struct DirectionBuffers {
  explicit DirectionBuffers(const DirectionPlan& plan)
      : slab_checks(plan.checks.size())
      , checks{plan.checks.size()}
      , verify{std::span<const VerifyCheck>{plan.verify}}
      , offsets{std::span<const int>{plan.offsets}}
      , verify_offsets{std::span<const int>{plan.verify_offsets}} {}

  std::vector<PlaneCheck> slab_checks;
  cuda::DeviceArray<PlaneCheck> checks;
  cuda::DeviceArray<VerifyCheck> verify;
  cuda::DeviceArray<int> offsets, verify_offsets;
  cuda::MatchBuffer matches;
};

// The two kernels compiled for one version.
struct Kernels {
  decltype(&fill_planes<Version::modern>) fill;
  decltype(&match_words<Version::modern>) match;
};

[[nodiscard]] Kernels kernels_for(Version version) {
  return with_version(version, [](auto tag) {
    return Kernels{fill_planes<decltype(tag)::value>, match_words<decltype(tag)::value>};
  });
}

// Everything one device needs to search the slabs of a plan: the field, the
// level table, and each direction's buffers. Lives on the device that was
// selected when it was created.
class DeviceSearch {
public:
  // nullopt if the device has no room for the field.
  [[nodiscard]] static std::optional<DeviceSearch> try_create(const SearchPlan& plan, Version version) {
    auto planes{cuda::DeviceArray<Planes>::try_allocate(plan.plane_words)};
    if (!planes)
      return std::nullopt;
    return DeviceSearch{plan, version, std::move(*planes)};
  }

  // Queues the fill of one slab, then each direction's match against it.
  void run(const Slab& slab) {
    const auto fill{kernels_.fill};
    const auto match{kernels_.match};
    if (slab.field_words > 0) {
      fill<<<(slab.field_words + THREADS - 1) / THREADS, THREADS>>>(slab.layout, plane_y_.get(),
                                                                    slab.field_words, planes_.get());
      cuda::check(cudaGetLastError());
    }
    for (std::size_t i{0}; i < buffers_.size(); ++i) {
      const auto& direction{plan_->directions[i]};
      auto& buffer{buffers_[i]};
      prepare_plane_checks(slab.layout, direction.checks, direction.offsets.data(), buffer.slab_checks);
      buffer.checks.upload(buffer.slab_checks);
      match<<<slab.match_blocks, THREADS>>>(slab.layout, buffer.checks.get(), buffer.offsets.get(),
                                            planes_.get(), buffer.verify.get(), buffer.verify_offsets.get(),
                                            buffer.matches.matches(), buffer.matches.count(), MAX_MATCHES);
      cuda::check(cudaGetLastError());
    }
  }

  // One result per direction, for every slab run so far.
  [[nodiscard]] std::vector<DirectionResult> collect() const {
    std::vector<DirectionResult> results;
    for (const auto& buffer : buffers_)
      results.push_back(buffer.matches.collect());
    return results;
  }

private:
  DeviceSearch(const SearchPlan& plan, Version version, cuda::DeviceArray<Planes> planes)
      : plan_{&plan}
      , kernels_{kernels_for(version)}
      , planes_{std::move(planes)}
      , plane_y_{std::span<const int>{plan.plane_y}} {
    buffers_.reserve(plan.directions.size());
    for (const auto& direction : plan.directions)
      buffers_.emplace_back(direction);
  }

  const SearchPlan* plan_;
  Kernels kernels_;
  cuda::DeviceArray<Planes> planes_;
  cuda::DeviceArray<int> plane_y_;
  std::vector<DirectionBuffers> buffers_;
};

// With several devices the slabs are kept short enough that each device gets
// about this many, so a faster device can take more than a slower one.
constexpr std::int64_t QUEUE_SLABS_PER_DEVICE{8};
constexpr std::int64_t MIN_QUEUE_SLAB_ROWS{64};

// The field budget every worker can meet. A device that is listed several
// times shares its free memory between its workers.
[[nodiscard]] std::size_t shared_plane_bytes(std::span<const int> devices, std::size_t directions) {
  auto budget{DEFAULT_PLANE_BYTES};
  for (const int device : devices) {
    cuda::check(cudaSetDevice(device));
    const auto workers{static_cast<std::size_t>(std::ranges::count(devices, device))};
    budget = std::min(budget, available_plane_bytes(directions * workers) / workers);
  }
  return budget;
}

// Each device takes the next slab nobody has started, searches it, and comes
// back for another, so the work divides by how fast each device is. A slab
// belongs to exactly one device, and its field covers the formation's reach
// past its own rows, so nothing depends on the neighbouring slab.
[[nodiscard]] std::optional<SharedSearchResult>
search_on_queue(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations,
                Version version, std::span<const int> devices) {
  const auto nz{static_cast<std::int64_t>(bounds.nz())};
  const auto wanted_slabs{static_cast<std::int64_t>(devices.size()) * QUEUE_SLABS_PER_DEVICE};
  const auto slab_rows{std::clamp<std::int64_t>(nz / wanted_slabs, MIN_QUEUE_SLAB_ROWS, DEFAULT_SLAB_ROWS)};
  const auto plan{plan_search(bounds, formations, shared_plane_bytes(devices, formations.size()), slab_rows)};
  if (!plan)
    return std::nullopt;

  const auto slabs{(nz + plan->slab_rows - 1) / plan->slab_rows};
  const auto row_positions{bounds.nx() * bounds.ny() * formations.size()};
  std::atomic<std::int64_t> next_slab{0};
  std::vector<std::vector<DirectionResult>> found(devices.size());

  const auto started{std::chrono::steady_clock::now()};
  cuda::on_each_device(devices, [&](std::size_t worker) {
    auto device{DeviceSearch::try_create(*plan, version)};
    if (!device)
      return; // the other devices take its share
    std::uint64_t rows{0};
    for (;;) {
      const auto index{next_slab.fetch_add(1)};
      if (index >= slabs)
        break;
      const auto slab{plan->slab(index * plan->slab_rows)};
      device->run(slab);
      // The slab is finished before this device asks for another.
      cuda::check(cudaDeviceSynchronize());
      rows += static_cast<std::uint64_t>(slab.layout.nz);
    }
    found[worker] = device->collect();
    cuda::add_work(worker, rows * row_positions);
  });
  const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - started};

  // No device had room for the field, so no slab was searched.
  if (std::ranges::all_of(found, [](const auto& results) { return results.empty(); }))
    return std::nullopt;

  SharedSearchResult output;
  output.kernel_seconds = elapsed.count();
  for (std::size_t direction{0}; direction < formations.size(); ++direction) {
    std::vector<DirectionResult> parts;
    for (auto& results : found) {
      if (!results.empty())
        parts.push_back(std::move(results[direction]));
    }
    output.directions.push_back(cuda::merge_results(std::move(parts)));
  }
  return output;
}

} // namespace

std::optional<SharedSearchResult>
search(const SearchBounds& bounds, std::span<const std::vector<BlockInfo>> formations, Version version) {
  const auto positions{bounds.nx() * bounds.ny() * bounds.nz() * formations.size()};
  const auto devices{cuda::devices_for(cuda::Path::bitplanes, positions)};
  if (devices.size() > 1) {
    auto output{search_on_queue(bounds, formations, version, devices)};
    if (output)
      cuda::note_rate(cuda::Path::bitplanes, positions, output->kernel_seconds, devices.size());
    return output;
  }

  cuda::check(cudaSetDevice(devices.front()));
  const auto plan{plan_search(bounds, formations, available_plane_bytes(formations.size()))};
  if (!plan)
    return std::nullopt;
  auto device{DeviceSearch::try_create(*plan, version)};
  if (!device)
    return std::nullopt;

  // Copies and kernels are ordered on the default stream.
  cuda::Timer timer;
  const auto nz{static_cast<std::int64_t>(bounds.nz())};
  for (std::int64_t z_offset{0}; z_offset < nz; z_offset += plan->slab_rows)
    device->run(plan->slab(z_offset));

  SharedSearchResult output;
  output.kernel_seconds = timer.stop_seconds();
  output.directions = device->collect();
  cuda::note_rate(cuda::Path::bitplanes, positions, output.kernel_seconds, 1);
  return output;
}

} // namespace cuda_bitplanes
