#include "backends/cuda/bitplane_search.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string_view>

#include "backends/cuda/bitplane_plan.hpp"
#include "backends/cuda/bitplanes.cuh"
#include "backends/cuda/device.hpp"
#include "backends/cuda/match_buffer.hpp"

namespace cuda_bitplanes {

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

} // namespace

bool requested() {
  const auto* setting{std::getenv("TF_CUDA_BITPLANES")};
  return setting && std::string_view{setting} == "1";
}

bool trace_enabled() {
  return std::getenv("TF_CUDA_BITPLANES_TRACE") != nullptr;
}

std::optional<SearchOutput> search(const SearchBounds& bounds,
                                   std::span<const std::vector<BlockInfo>> formations) {
  const auto plan{plan_search(bounds, formations, available_plane_bytes(formations.size()))};
  if (!plan)
    return std::nullopt;
  const auto planes{cuda::DeviceArray<Planes>::try_allocate(plan->plane_words)};
  if (!planes)
    return std::nullopt;

  const cuda::DeviceArray<int> plane_y{std::span<const int>{plan->plane_y}};
  std::vector<DirectionBuffers> buffers;
  buffers.reserve(plan->directions.size());
  for (const auto& direction : plan->directions)
    buffers.emplace_back(direction);

  // Copies and kernels are ordered on the default stream.
  cuda::Timer timer;
  const auto nz{static_cast<std::int64_t>(bounds.nz())};
  for (std::int64_t z_offset{0}; z_offset < nz;) {
    const auto slab{plan->slab(z_offset)};
    if (slab.field_words > 0) {
      fill_planes<true, true><<<(slab.field_words + THREADS - 1) / THREADS, THREADS>>>(
          slab.layout, plane_y.get(), slab.field_words, planes->get());
      cuda::check(cudaGetLastError());
    }
    for (std::size_t i{0}; i < buffers.size(); ++i) {
      const auto& direction{plan->directions[i]};
      auto& buffer{buffers[i]};
      prepare_plane_checks(slab.layout, direction.checks, direction.offsets.data(), buffer.slab_checks);
      buffer.checks.upload(buffer.slab_checks);
      match_words<MatchMode::Search, true, true><<<slab.match_blocks, THREADS>>>(
          slab.layout, buffer.checks.get(), buffer.offsets.get(), planes->get(), nullptr, nullptr, 0,
          buffer.verify.get(), buffer.verify_offsets.get(), buffer.matches.matches(), buffer.matches.count(),
          MAX_MATCHES);
      cuda::check(cudaGetLastError());
    }
    z_offset += slab.layout.nz;
  }

  SearchOutput output{.kernel_seconds = timer.stop_seconds()};
  for (const auto& buffer : buffers)
    output.directions.push_back(buffer.matches.collect());
  return output;
}

} // namespace cuda_bitplanes
