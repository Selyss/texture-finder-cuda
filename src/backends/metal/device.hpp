#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>

#include "search/backend.hpp"

// The Metal device, its command queue and the compiled kernels, shared by the
// searches of this backend, and the buffers a kernel reports matches through.
namespace metal {

[[noreturn]] void fail(const std::string& what, NSError* error);

// Throws if the machine has no Metal device.
[[nodiscard]] id<MTLDevice> device();
[[nodiscard]] id<MTLCommandQueue> queue();

// The kernel `name` specialised for `version`. Compiling the shader costs
// about 100 ms, so it is compiled once and every pipeline is kept.
[[nodiscard]] id<MTLComputePipelineState> pipeline(NSString* name, Version version);

// Zeroed memory that the host and the kernels both see.
[[nodiscard]] id<MTLBuffer> shared_buffer(std::size_t bytes);

template <class T>
[[nodiscard]] id<MTLBuffer> shared_buffer(std::span<const T> values) {
  id<MTLBuffer> buffer = shared_buffer(values.size_bytes());
  std::memcpy(buffer.contents, values.data(), values.size_bytes());
  return buffer;
}

// The value at the start of a shared buffer.
template <class T>
[[nodiscard]] T& contents_as(id<MTLBuffer> buffer) {
  return *static_cast<T*>(buffer.contents);
}

// Dispatches whole threadgroups that cover a grid of `x` by `y` by `z`
// threads. The kernels return on the ragged edge.
void dispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger x,
              NSUInteger y, NSUInteger z);

// Commits the commands, waits for them and returns their GPU seconds.
double run(id<MTLCommandBuffer> commands);

// Where a kernel reports matches: room for the first MAX_MATCHES of them, how
// many of those are stored, and a count of every match in the current pass.
// That count is 32 bits, so a pass covers fewer than 2^32 positions.
class MatchBuffer {
public:
  MatchBuffer();

  // Binds the three buffers at `index`, `index + 1` and `index + 2`.
  void bind(id<MTLComputeCommandEncoder> encoder, NSUInteger index) const;

  // Adds the finished pass to the total. Call after each run that used the buffer.
  void finish_pass();

  // The matches sorted by x, y, z. The time is left for the caller.
  [[nodiscard]] DirectionResult collect() const;

private:
  id<MTLBuffer> matches_, stored_, pass_count_;
  std::uint64_t total_{0};
};

} // namespace metal
