#include "backends/metal/device.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace metal {

namespace {

// texture_rules.hpp, the bitplane headers and the kernels as one source
// string, pasted together by the Makefile.
const char shader_body[] =
#include "tf_shader_src.h"
    ;

id<MTLDevice> s_device;
id<MTLCommandQueue> s_queue;
id<MTLLibrary> s_library;
NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* s_pipelines;

[[nodiscard]] id<MTLLibrary> library() {
  if (!s_library) {
    const auto source{"#define TF_MAX_MATCHES " + std::to_string(MAX_MATCHES) + "u\n" + shader_body};
    NSError* error = nil;
    s_library = [device() newLibraryWithSource:@(source.c_str()) options:nil error:&error];
    if (!s_library)
      fail("shader compilation failed", error);
  }
  return s_library;
}

} // namespace

void fail(const std::string& what, NSError* error) {
  throw std::runtime_error{"Metal error: " + what +
                           (error ? std::string{": "} + error.localizedDescription.UTF8String : "")};
}

id<MTLDevice> device() {
  if (!s_device) {
    s_device = MTLCreateSystemDefaultDevice();
    if (!s_device)
      throw std::runtime_error{"No Metal device available; rebuild with `make BACKEND=cpu`"};
  }
  return s_device;
}

id<MTLCommandQueue> queue() {
  if (!s_queue)
    s_queue = [device() newCommandQueue];
  return s_queue;
}

id<MTLComputePipelineState> pipeline(NSString* name, Version version) {
  // Branches on the version are resolved when the pipeline is created.
  const auto index{static_cast<int>(version)};
  NSString* key = [NSString stringWithFormat:@"%@/%d", name, index];
  if (id<MTLComputePipelineState> known = s_pipelines[key])
    return known;

  MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
  [constants setConstantValue:&index type:MTLDataTypeInt atIndex:0];
  NSError* error = nil;
  id<MTLFunction> function = [library() newFunctionWithName:name constantValues:constants error:&error];
  if (!function)
    fail("function specialization failed", error);
  id<MTLComputePipelineState> created = [device() newComputePipelineStateWithFunction:function error:&error];
  if (!created)
    fail("pipeline creation failed", error);
  if (!s_pipelines)
    s_pipelines = [NSMutableDictionary new];
  s_pipelines[key] = created;
  return created;
}

id<MTLBuffer> shared_buffer(std::size_t bytes) {
  // Metal has no empty buffers.
  id<MTLBuffer> buffer = [device() newBufferWithLength:std::max<std::size_t>(bytes, 1)
                                               options:MTLResourceStorageModeShared];
  if (!buffer)
    fail("buffer allocation failed", nil);
  return buffer;
}

void dispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger x,
              NSUInteger y, NSUInteger z) {
  const NSUInteger width{pipeline.threadExecutionWidth};
  const NSUInteger limit{std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup)};
  const MTLSize group{MTLSizeMake(width, 1, std::max<NSUInteger>(1, limit / width))};
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreadgroups:MTLSizeMake((x + group.width - 1) / group.width,
                                            (y + group.height - 1) / group.height,
                                            (z + group.depth - 1) / group.depth)
          threadsPerThreadgroup:group];
}

double run(id<MTLCommandBuffer> commands) {
  [commands commit];
  [commands waitUntilCompleted];
  if (commands.error)
    fail("command buffer failed", commands.error);
  return commands.GPUEndTime - commands.GPUStartTime;
}

MatchBuffer::MatchBuffer()
    : matches_{shared_buffer(MAX_MATCHES * sizeof(MatchResult))}
    , stored_{shared_buffer(sizeof(unsigned int))}
    , pass_count_{shared_buffer(sizeof(unsigned int))} {}

void MatchBuffer::bind(id<MTLComputeCommandEncoder> encoder, NSUInteger index) const {
  [encoder setBuffer:matches_ offset:0 atIndex:index];
  [encoder setBuffer:stored_ offset:0 atIndex:index + 1];
  [encoder setBuffer:pass_count_ offset:0 atIndex:index + 2];
}

void MatchBuffer::finish_pass() {
  total_ += std::exchange(contents_as<unsigned int>(pass_count_), 0u);
}

DirectionResult MatchBuffer::collect() const {
  const auto stored{std::min(contents_as<unsigned int>(stored_), MAX_MATCHES)};
  const auto* first{static_cast<const MatchResult*>(matches_.contents)};
  DirectionResult result{
      .matches = {first, first + stored}, .total = total_, .truncated = total_ > MAX_MATCHES};
  std::ranges::sort(result.matches, {}, [](const MatchResult& m) { return std::tuple{m.x, m.y, m.z}; });
  return result;
}

} // namespace metal
