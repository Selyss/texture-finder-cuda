// Apple GPU backend via Metal. The compute shader is built at runtime from
// a source string generated at build time by concatenating
// src/texture/texture_rules.hpp, the bitplane headers and the kernels (see
// the Makefile), so the RNG is the same single source every other backend
// compiles. The plain kernel in match.metal tests every position against
// every block. Formations that filter well are searched with bitplanes
// instead (bitplane_search.hpp), which must return the same matches.

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "backends/metal/bitplane_search.hpp"
#include "backends/metal/device.hpp"
#include "bitplanes/shared_search.hpp"
#include "search/backend.hpp"

std::string_view search_backend_name() {
  return "Metal";
}

namespace {

struct TfParams {
  int x_min, x_max;
  int y_min, y_max;
  int z_max;  // last z of this dispatch
  int z_base; // first z of this dispatch
  int total_blocks;
  int pad;
};

struct TfBlock {
  int x, y, z, w;
};

[[nodiscard]] TfBlock shader_block(const BlockInfo& block) {
  return {.x = block.x, .y = block.y, .z = block.z, .w = block.rotation | ((block.is_side ? 1 : 3) << 8)};
}

// Work is dispatched in z slices to keep individual command buffers short.
[[nodiscard]] DirectionResult plain_search(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks,
                                           Version version) {
  std::vector<TfBlock> shader_blocks;
  for (const auto& block : blocks)
    shader_blocks.push_back(shader_block(block));

  DirectionResult result;
  @autoreleasepool {
    id<MTLComputePipelineState> pipeline = metal::pipeline(@"match_formation", version);
    id<MTLBuffer> block_buffer = metal::shared_buffer(std::span<const TfBlock>{shader_blocks});
    metal::MatchBuffer matches;

    // z layers per dispatch, chosen so one dispatch stays around a billion threads.
    const auto nx{bounds.nx()}, ny{bounds.ny()}, nz{bounds.nz()};
    const auto z_step{std::clamp<std::uint64_t>((std::uint64_t{1} << 30) / (nx * ny), 1, nz)};

    double seconds{0.0};
    for (std::uint64_t z_done{0}; z_done < nz; z_done += z_step) {
      const auto z_count{std::min(z_step, nz - z_done)};
      // The grid is rounded up to whole threadgroups, so z_max is this
      // dispatch's own last row. The rows after it belong to the next one.
      const auto z_base{bounds.z_min + static_cast<std::int64_t>(z_done)};
      const TfParams params{.x_min = bounds.x_min,
                            .x_max = bounds.x_max,
                            .y_min = bounds.y_min,
                            .y_max = bounds.y_max,
                            .z_max = static_cast<int>(z_base + static_cast<std::int64_t>(z_count) - 1),
                            .z_base = static_cast<int>(z_base),
                            .total_blocks = static_cast<int>(shader_blocks.size()),
                            .pad = 0};

      id<MTLCommandBuffer> commands = [metal::queue() commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      [encoder setBuffer:block_buffer offset:0 atIndex:0];
      [encoder setBytes:&params length:sizeof(params) atIndex:1];
      matches.bind(encoder, 2);
      metal::dispatch(encoder, pipeline, nx, ny, z_count);
      [encoder endEncoding];
      seconds += metal::run(commands);
      matches.finish_pass();
    }

    result = matches.collect();
    result.kernel_seconds = seconds;
  }
  return result;
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

  return bitplanes::search_one(
      blocks, version,
      [&](std::span<const std::vector<BlockInfo>> formations) {
        return metal_bitplanes::search(bounds, formations, version);
      },
      [&] { return plain_search(bounds, blocks, version); });
}

// On unless TF_METAL_BITPLANES=0, which keeps every search on the plain kernel.
bool shared_search_enabled(Version) {
  return bitplanes::enabled_by("TF_METAL_BITPLANES");
}

SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                     std::span<const std::vector<BlockInfo>> directions, Version version) {
  return bitplanes::search_sharing_field(bounds, directions, version,
                                         [&](std::span<const std::vector<BlockInfo>> formations) {
                                           return metal_bitplanes::search(bounds, formations, version);
                                         });
}
