#include "backends/metal/bitplane_search.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#include "backends/metal/device.hpp"
#include "bitplanes/plan.hpp"

namespace metal_bitplanes {

using namespace bitplanes;

namespace {

constexpr std::size_t MAX_PLANE_BYTES{std::size_t{1} << 30};

// The kernels count a pass's matches in 32 bits, so a slab has no more
// candidates than that can count.
constexpr std::uint64_t MAX_SLAB_CANDIDATES{std::numeric_limits<std::uint32_t>::max()};

[[nodiscard]] std::int64_t max_slab_rows(const SearchBounds& bounds) {
  const auto nx{bounds.nx()}, ny{bounds.ny()};
  if (nx == 0 || ny == 0)
    return 0;
  return static_cast<std::int64_t>(std::min<std::uint64_t>(DEFAULT_SLAB_ROWS, MAX_SLAB_CANDIDATES / nx / ny));
}

// The field lives on the GPU only. It is kept between searches, because a
// long search calls in once per slice.
id<MTLBuffer> s_planes;

[[nodiscard]] id<MTLBuffer> field_buffer(std::size_t bytes) {
  if (!s_planes || s_planes.length < bytes)
    s_planes = [metal::device() newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  return s_planes;
}

// One direction's buffers. The plane checks are rewritten per slab.
struct DirectionBuffers {
  explicit DirectionBuffers(const DirectionPlan& plan)
      : slab_checks(plan.checks.size())
      , checks{metal::shared_buffer(plan.checks.size() * sizeof(PlaneCheck))}
      , verify{metal::shared_buffer(std::span<const VerifyCheck>{plan.verify})}
      , offsets{metal::shared_buffer(std::span<const int>{plan.offsets})}
      , verify_offsets{metal::shared_buffer(std::span<const int>{plan.verify_offsets})} {}

  std::vector<PlaneCheck> slab_checks;
  id<MTLBuffer> checks, verify, offsets, verify_offsets;
  metal::MatchBuffer matches;
};

} // namespace

bool enabled() {
  const auto* setting{std::getenv("TF_METAL_BITPLANES")};
  return !setting || std::string_view{setting} != "0";
}

bool trace_enabled() {
  return std::getenv("TF_METAL_BITPLANES_TRACE") != nullptr;
}

std::optional<SearchOutput> search(const SearchBounds& bounds,
                                   std::span<const std::vector<BlockInfo>> formations, Version version) {
  const auto slab_rows{max_slab_rows(bounds)};
  if (slab_rows == 0)
    return std::nullopt;

  SearchOutput output;
  @autoreleasepool {
    const auto plane_bytes{std::min<std::size_t>(MAX_PLANE_BYTES, metal::device().maxBufferLength)};
    const auto plan{plan_search(bounds, formations, plane_bytes, slab_rows)};
    if (!plan)
      return std::nullopt;
    id<MTLBuffer> planes = field_buffer(plan->plane_words * sizeof(Planes));
    if (!planes)
      return std::nullopt;

    id<MTLComputePipelineState> fill = metal::pipeline(@"fill_planes", version);
    id<MTLComputePipelineState> match = metal::pipeline(@"match_words", version);
    id<MTLBuffer> plane_y = metal::shared_buffer(std::span<const int>{plan->plane_y});
    std::vector<DirectionBuffers> buffers;
    buffers.reserve(plan->directions.size());
    for (const auto& direction : plan->directions)
      buffers.emplace_back(direction);

    // One command buffer per slab: the fill, then each direction's match.
    const auto nz{static_cast<std::int64_t>(bounds.nz())};
    for (std::int64_t z_offset{0}; z_offset < nz;) {
      const auto slab{plan->slab(z_offset)};
      const auto& layout{slab.layout};
      id<MTLCommandBuffer> commands = [metal::queue() commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      [encoder setBytes:&layout length:sizeof(layout) atIndex:0];
      if (slab.field_words > 0) {
        [encoder setBuffer:plane_y offset:0 atIndex:1];
        [encoder setBuffer:planes offset:0 atIndex:2];
        metal::dispatch(encoder, fill, layout.field_words, layout.hashed_rows, layout.y_slots);
      }
      for (std::size_t i{0}; i < buffers.size(); ++i) {
        const auto& direction{plan->directions[i]};
        auto& buffer{buffers[i]};
        prepare_plane_checks(layout, direction.checks, direction.offsets.data(), buffer.slab_checks);
        std::memcpy(buffer.checks.contents, buffer.slab_checks.data(),
                    buffer.slab_checks.size() * sizeof(PlaneCheck));
        [encoder setBuffer:buffer.checks offset:0 atIndex:1];
        [encoder setBuffer:buffer.offsets offset:0 atIndex:2];
        [encoder setBuffer:planes offset:0 atIndex:3];
        [encoder setBuffer:buffer.verify offset:0 atIndex:4];
        [encoder setBuffer:buffer.verify_offsets offset:0 atIndex:5];
        buffer.matches.bind(encoder, 6);
        metal::dispatch(encoder, match, layout.candidate_words, layout.ny, layout.nz);
      }
      [encoder endEncoding];
      output.kernel_seconds += metal::run(commands);
      for (auto& buffer : buffers)
        buffer.matches.finish_pass();
      z_offset += layout.nz;
    }

    for (const auto& buffer : buffers)
      output.directions.push_back(buffer.matches.collect());
  }
  return output;
}

} // namespace metal_bitplanes
