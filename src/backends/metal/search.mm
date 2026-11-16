// Apple GPU backend via Metal. The compute shader is built at runtime from
// a source string generated at build time by concatenating
// src/texture/texture_rules.hpp and src/backends/metal/match.metal (see the Makefile), so the RNG is
// the same single source every other backend compiles. Work is dispatched
// in z-slices to keep individual command buffers short.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
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

[[noreturn]] void fail(const char* what, NSError* err) {
  std::fprintf(stderr, "Metal error: %s%s%s\n", what, err ? ": " : "",
               err ? err.localizedDescription.UTF8String : "");
  std::exit(2);
}

} // namespace

namespace {

// Pipeline cache: run_search is called once per slice by the host-side
// progress/checkpoint driver, and shader compilation costs ~100 ms — done
// once per version instead of once per call.
id<MTLDevice> s_device;
id<MTLCommandQueue> s_queue;
id<MTLComputePipelineState> s_pso[NUM_VERSIONS];

id<MTLComputePipelineState> ensure_pipeline(int version) {
  if (s_pso[version])
    return s_pso[version];

  if (!s_device) {
    s_device = MTLCreateSystemDefaultDevice();
    if (!s_device) {
      std::fprintf(stderr, "No Metal device available; rebuild with `make BACKEND=cpu`\n");
      std::exit(2);
    }
    s_queue = [s_device newCommandQueue];
  }

  std::string src = "#define TF_MAX_MATCHES " + std::to_string(MAX_MATCHES) + "u\n";
  src += shader_body;

  NSError* err = nil;
  id<MTLLibrary> lib = [s_device newLibraryWithSource:@(src.c_str()) options:nil error:&err];
  if (!lib)
    fail("shader compilation failed", err);

  MTLFunctionConstantValues* consts = [MTLFunctionConstantValues new];
  int version_const = version;
  [consts setConstantValue:&version_const type:MTLDataTypeInt atIndex:0];
  id<MTLFunction> fn = [lib newFunctionWithName:@"match_formation" constantValues:consts error:&err];
  if (!fn)
    fail("function specialization failed", err);
  id<MTLComputePipelineState> pso = [s_device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso)
    fail("pipeline creation failed", err);
  s_pso[version] = pso;
  return pso;
}

} // namespace

DirectionResult run_search(const SearchBounds& bounds, const SearchFormation& formation, int version) {
  const auto& tops_and_bottoms{formation.tops};
  const auto& sides{formation.sides};
  if (tops_and_bottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS) {
    std::fprintf(stderr, "Formation exceeds %d blocks per face type\n", MAX_FORMATION_BLOCKS);
    std::exit(1);
  }

  std::vector<TfBlock> blocks;
  blocks.reserve(tops_and_bottoms.size() + sides.size());
  for (const auto& t : tops_and_bottoms)
    blocks.push_back({t.x, t.y, t.z, t.rotation | (3 << 8)});
  for (const auto& s : sides)
    blocks.push_back({s.x, s.y, s.z, s.rotation | (1 << 8)});
  if (blocks.empty()) {
    std::fprintf(stderr, "Empty formation\n");
    std::exit(1);
  }

  const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
  const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
  const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
  if (ny > (~0ull) / nz || nx > (~0ull) / (ny * nz)) {
    std::fprintf(stderr, "Search volume too large\n");
    std::exit(1);
  }

  std::vector<MatchResult> matches;
  unsigned long long total = 0;
  float ms = 0.f;

  @autoreleasepool {
    id<MTLComputePipelineState> pso = ensure_pipeline(version);
    id<MTLDevice> device = s_device;
    id<MTLCommandQueue> queue = s_queue;

    id<MTLBuffer> b_blocks = [device newBufferWithBytes:blocks.data()
                                                 length:blocks.size() * sizeof(TfBlock)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_out = [device newBufferWithLength:MAX_MATCHES * sizeof(MatchResult)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_write_idx = [device newBufferWithLength:sizeof(unsigned int)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_slice = [device newBufferWithLength:sizeof(unsigned int)
                                                options:MTLResourceStorageModeShared];
    *(unsigned int*)b_write_idx.contents = 0;

    // Slice depth chosen so one dispatch stays around a billion threads.
    const unsigned long long per_layer = nx * ny;
    unsigned long long z_step = per_layer > 0 ? (1ull << 30) / per_layer : nz;
    if (z_step == 0)
      z_step = 1;
    if (z_step > nz)
      z_step = nz;

    const NSUInteger tew = pso.threadExecutionWidth;
    const NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
    const MTLSize tg = MTLSizeMake(tew, 1, std::max<NSUInteger>(1, std::min<NSUInteger>(256, max_tg) / tew));

    for (unsigned long long z0 = 0; z0 < nz; z0 += z_step) {
      const unsigned long long z_count = std::min(z_step, nz - z0);
      TfParams P = {bounds.x_min,       bounds.x_max,
                    bounds.y_min,       bounds.y_max,
                    bounds.z_max,       (int)(bounds.z_min + (long long)z0),
                    (int)blocks.size(), 0};
      *(unsigned int*)b_slice.contents = 0;

      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:b_blocks offset:0 atIndex:0];
      [enc setBytes:&P length:sizeof(P) atIndex:1];
      [enc setBuffer:b_out offset:0 atIndex:2];
      [enc setBuffer:b_write_idx offset:0 atIndex:3];
      [enc setBuffer:b_slice offset:0 atIndex:4];
      const MTLSize grid = MTLSizeMake((nx + tg.width - 1) / tg.width, (ny + tg.height - 1) / tg.height,
                                       (z_count + tg.depth - 1) / tg.depth);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error)
        fail("command buffer failed", cb.error);
      ms += (float)((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
      total += *(unsigned int*)b_slice.contents;
    }

    const unsigned int wrote = *(unsigned int*)b_write_idx.contents;
    const unsigned int stored = std::min(wrote, MAX_MATCHES);
    matches.assign((MatchResult*)b_out.contents, (MatchResult*)b_out.contents + stored);
  }

  std::sort(matches.begin(), matches.end(), [](const MatchResult& a, const MatchResult& b) {
    return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z);
  });
  return {std::move(matches), total, total > MAX_MATCHES, ms / 1000.0};
}

bool shared_search_enabled(int) {
  return false;
}

SharedSearchResult run_shared_search(const SearchBounds& bounds, std::span<const SearchFormation> formations,
                                     int version) {
  SharedSearchResult result;
  result.directions.resize(formations.size());
  for (size_t i = 0; i < formations.size(); ++i) {
    result.directions[i] = run_search(bounds, formations[i], version);
    result.kernel_seconds += result.directions[i].kernel_seconds;
  }
  return result;
}
