// Apple GPU backend via Metal. The compute shader is built at runtime from
// a source string generated at build time by concatenating
// src/texture/texture_rules.hpp and src/backends/metal/match.metal (see the Makefile), so the RNG is
// the same single source every other backend compiles. Work is dispatched
// in z-slices to keep individual command buffers short.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "search/backend.hpp"

std::string_view search_backend_name() {
  return "Metal";
}

namespace {

const char shader_body[] =
#include "tf_shader_src.h"
    ;

struct TfParams {
  int x_min, x_max;
  int y_min, y_max;
  int z_max;
  int z_base;
  int total_blocks;
  int pad;
};

struct TfBlock {
  int x, y, z, w;
};

[[noreturn]] void fail(const std::string& what, NSError* error) {
  throw std::runtime_error{"Metal error: " + what +
                           (error ? std::string{": "} + error.localizedDescription.UTF8String : "")};
}

// Pipeline cache: run_search is called once per slice by the host-side
// progress/checkpoint driver, and shader compilation costs ~100 ms — done
// once per version instead of once per call.
id<MTLDevice> s_device;
id<MTLCommandQueue> s_queue;
id<MTLComputePipelineState> s_pso[NUM_VERSIONS];

id<MTLComputePipelineState> ensure_pipeline(Version version) {
  const auto index{static_cast<int>(version)};
  if (s_pso[index])
    return s_pso[index];

  if (!s_device) {
    s_device = MTLCreateSystemDefaultDevice();
    if (!s_device)
      throw std::runtime_error{"No Metal device available; rebuild with `make BACKEND=cpu`"};
    s_queue = [s_device newCommandQueue];
  }

  std::string src = "#define TF_MAX_MATCHES " + std::to_string(MAX_MATCHES) + "u\n";
  src += shader_body;

  NSError* err = nil;
  id<MTLLibrary> lib = [s_device newLibraryWithSource:@(src.c_str()) options:nil error:&err];
  if (!lib)
    fail("shader compilation failed", err);

  MTLFunctionConstantValues* consts = [MTLFunctionConstantValues new];
  [consts setConstantValue:&index type:MTLDataTypeInt atIndex:0];
  id<MTLFunction> fn = [lib newFunctionWithName:@"match_formation" constantValues:consts error:&err];
  if (!fn)
    fail("function specialization failed", err);
  id<MTLComputePipelineState> pso = [s_device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso)
    fail("pipeline creation failed", err);
  s_pso[index] = pso;
  return pso;
}

// One shared-storage buffer holding a single value the shader updates.
template <class T>
[[nodiscard]] T& contents_as(id<MTLBuffer> buffer) {
  return *static_cast<T*>(buffer.contents);
}

[[nodiscard]] TfBlock shader_block(const BlockInfo& block) {
  return {.x = block.x, .y = block.y, .z = block.z, .w = block.rotation | ((block.is_side ? 1 : 3) << 8)};
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

  std::vector<TfBlock> shader_blocks;
  for (const auto& block : blocks)
    shader_blocks.push_back(shader_block(block));

  DirectionResult result;
  @autoreleasepool {
    id<MTLComputePipelineState> pipeline = ensure_pipeline(version);
    const auto shared_buffer = [](std::size_t bytes) {
      return [s_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> block_buffer = [s_device newBufferWithBytes:shader_blocks.data()
                                                       length:shader_blocks.size() * sizeof(TfBlock)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> match_buffer = shared_buffer(MAX_MATCHES * sizeof(MatchResult));
    id<MTLBuffer> written_buffer = shared_buffer(sizeof(unsigned int));
    id<MTLBuffer> slice_count_buffer = shared_buffer(sizeof(unsigned int));
    contents_as<unsigned int>(written_buffer) = 0;

    // z layers per dispatch, chosen so one dispatch stays around a billion threads.
    const auto layer{nx * ny};
    const auto z_step{std::clamp<std::uint64_t>((std::uint64_t{1} << 30) / layer, 1, nz)};

    const NSUInteger width{pipeline.threadExecutionWidth};
    const NSUInteger group_limit{std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup)};
    const MTLSize group{MTLSizeMake(width, 1, std::max<NSUInteger>(1, group_limit / width))};

    for (std::uint64_t z_done{0}; z_done < nz; z_done += z_step) {
      const auto z_count{std::min(z_step, nz - z_done)};
      const TfParams params{
          .x_min = bounds.x_min,
          .x_max = bounds.x_max,
          .y_min = bounds.y_min,
          .y_max = bounds.y_max,
          // The grid is rounded up to whole threadgroups, so the kernel is
          // told where this dispatch ends, not where the search ends.
          .z_max = static_cast<int>(bounds.z_min + static_cast<std::int64_t>(z_done + z_count) - 1),
          .z_base = static_cast<int>(bounds.z_min + static_cast<std::int64_t>(z_done)),
          .total_blocks = static_cast<int>(shader_blocks.size()),
          .pad = 0};
      contents_as<unsigned int>(slice_count_buffer) = 0;

      id<MTLCommandBuffer> commands = [s_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:block_buffer offset:0 atIndex:0];
      [encoder setBytes:&params length:sizeof(params) atIndex:1];
      [encoder setBuffer:match_buffer offset:0 atIndex:2];
      [encoder setBuffer:written_buffer offset:0 atIndex:3];
      [encoder setBuffer:slice_count_buffer offset:0 atIndex:4];
      const MTLSize grid{MTLSizeMake((nx + group.width - 1) / group.width,
                                     (ny + group.height - 1) / group.height,
                                     (z_count + group.depth - 1) / group.depth)};
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
      [encoder endEncoding];
      [commands commit];
      [commands waitUntilCompleted];
      if (commands.error)
        fail("command buffer failed", commands.error);
      result.kernel_seconds += commands.GPUEndTime - commands.GPUStartTime;
      result.total += contents_as<unsigned int>(slice_count_buffer);
    }

    const auto stored{std::min(contents_as<unsigned int>(written_buffer), MAX_MATCHES)};
    const auto* first{static_cast<const MatchResult*>(match_buffer.contents)};
    result.matches.assign(first, first + stored);
  }

  result.truncated = result.total > MAX_MATCHES;
  std::ranges::sort(result.matches, {}, [](const MatchResult& m) { return std::tuple{m.x, m.y, m.z}; });
  return result;
}

bool shared_search_enabled(Version) {
  return false;
}

SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                     std::span<const std::vector<BlockInfo>> directions, Version version) {
  return search_each_direction(bounds, directions, version);
}
