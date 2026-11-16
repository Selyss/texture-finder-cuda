#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cuda/bitplane_search.hpp"
#include "backends/cuda/device.hpp"
#include "backends/cuda/match_buffer.hpp"
#include "bitplanes/shared_search.hpp"
#include "texture/texture_rules.hpp"

std::string_view search_backend_name() {
  return "CUDA";
}

// Tuning knobs, overridable at compile time (test/bench.sh sweeps these;
// defaults are the winners on an RTX 3090).
#ifndef TF_BLOCK_THREADS
#define TF_BLOCK_THREADS 256
#endif
// Positions each lane pulls per work chunk; one chunk = 32x this.
#ifndef TF_CHUNK_PER_LANE
#define TF_CHUNK_PER_LANE 1024
#endif

constexpr int WARP = 32;

static_assert(TF_CHUNK_PER_LANE > 0, "zero chunk size would livelock the chunk cursor");
static_assert(TF_BLOCK_THREADS % WARP == 0, "partial warps break the full-mask warp intrinsics");

// One formation block with its hash terms precomputed: the position hash's
// x and z multiplies distribute over the block offset under wrapping (see
// coord_random_from_parts), so per evaluation the kernel only adds these to the
// candidate's own terms instead of multiplying.
// w = expected rotation | (compare mask << 8); mask is 3 for top faces and 1
// for side faces, because for mods 2 and 4 both generators reduce to the
// mod-4 value masked down (see texture/texture_rules.hpp: side value == top roll % 2).
struct DevBlock {
  long long z_term;    // bz * 116129781
  unsigned int x_term; // bx * 3129871 (mod 2^32)
  int dy;
  int w;
};

__constant__ DevBlock d_blocks[2 * MAX_FORMATION_BLOCKS];

// A maximum-size formation must fit the default dynamic shared memory limit.
static_assert(2 * MAX_FORMATION_BLOCKS * sizeof(DevBlock) <= 48 * 1024,
              "formation no longer fits default dynamic shared memory");

// Warp-compacted search. The naive one-thread-per-position kernel wastes ~2/3
// of the issue slots: with a 1-in-4 pass rate per check, a warp's slowest lane
// dictates ~4.3 evaluations per batch while the average position needs ~1.33.
// Here every lane keeps its own candidate position and block cursor; the
// moment a candidate fails, the lane pulls a fresh position from its stripe
// of the current warp chunk. Lanes therefore stay busy evaluating block 0/1
// almost all the time. Warps grab chunks from a global cursor; within a chunk
// lane L owns linear positions chunk_base + pull*32 + L, so every position in
// [0, total) is evaluated exactly once, no matter how the grid is sized.
template <Version VERSION>
__global__ void __launch_bounds__(TF_BLOCK_THREADS)
    match_formation_kernel(SearchBounds b, int total_blocks, unsigned long long total,
                           unsigned long long* chunk_cursor, MatchResult* out,
                           unsigned long long* out_count) {
  // Formation goes to shared memory: compacted lanes sit at different block
  // indices, and divergent-address reads serialize in constant memory but
  // not in shared memory. Dynamically sized so small formations cost no
  // occupancy.
  extern __shared__ DevBlock s_blocks[];
  for (int i = (int)threadIdx.x; i < total_blocks; i += (int)blockDim.x)
    s_blocks[i] = d_blocks[i];
  __syncthreads();

  const int lane = threadIdx.x & (WARP - 1);
  const long long nx = (long long)b.x_max - b.x_min + 1;
  const long long nz = (long long)b.z_max - b.z_min + 1;
  const unsigned long long chunk_size = (unsigned long long)WARP * TF_CHUNK_PER_LANE;

  while (true) {
    // The warp takes a chunk; lane L owns linear positions
    // base + pull*32 + L, so every index in [0, total) is evaluated
    // exactly once across all warps.
    unsigned long long base = 0;
    if (lane == 0)
      base = atomicAdd(chunk_cursor, chunk_size);
    base = __shfl_sync(0xffffffffu, base, 0);
    if (base >= total)
      return; // uniform exit

    // Number of in-range pulls in this lane's stripe, computed once so
    // the per-pull path needs no 64-bit index math or range compare:
    // positions g_lane, g_lane+32, ..., g_lane+(budget-1)*32 are < total.
    const unsigned long long g_lane = base + lane;
    int budget = 0;
    if (g_lane < total) {
      const unsigned long long remaining = total - g_lane;
      const unsigned long long pulls = (remaining + WARP - 1) / WARP;
      budget = (int)(pulls < TF_CHUNK_PER_LANE ? pulls : TF_CHUNK_PER_LANE);
    }

    int x = 0, y = 0, z = 0; // candidate origin (valid while bi >= 0)
    unsigned int hx = 0;     // x * 3129871 (mod 2^32)
    long long hz = 0;        // z * 116129781
    int bi = -1;             // next block to check; -1 = no live candidate
    bool first = true;

    // Each lane processes its stripe independently: the moment a
    // candidate fails, the lane pulls a fresh one, so lanes stay busy on
    // the highly selective first blocks instead of idling while a
    // warp-mate finishes a deep near-match.
    while (bi >= 0 || budget > 0) {
      if (bi < 0) {
        budget--;
        if (first) {
          // First pull of a chunk: full decompose (amortized away).
          first = false;
          unsigned long long r = g_lane;
          x = b.x_min + (int)(r % (unsigned long long)nx);
          r /= (unsigned long long)nx;
          z = b.z_min + (int)(r % (unsigned long long)nz);
          r /= (unsigned long long)nz;
          y = b.y_min + (int)r;
          hx = (unsigned int)x * 3129871u;
          hz = (long long)z * 116129781LL;
        } else {
          // Consecutive pulls advance exactly WARP linear steps;
          // the x hash term advances by a compile-time constant
          // (wrapping add), so the common path multiplies nothing.
          x += WARP;
          hx += (unsigned int)WARP * 3129871u;
          if (x > b.x_max) {
            do {
              x -= (int)nx;
              z++;
              if (z > b.z_max) {
                z -= (int)nz;
                y++;
              }
            } while (x > b.x_max);
            hx = (unsigned int)x * 3129871u;
            hz = (long long)z * 116129781LL;
          }
        }
        bi = 0;
      }

      const DevBlock blk = s_blocks[bi];
      const int expected = blk.w & 0xff;
      const int mask = blk.w >> 8;
      const int64_t mix = coord_mix_from_parts(hx + blk.x_term, hz + blk.z_term, y + blk.dy);
      const int v = texture_from_mix(VERSION, mix, 4);
      if ((v & mask) != expected) {
        bi = -1;
      } else if (++bi == total_blocks) {
        // 64-bit: a dense weak-formation search can exceed 2^32
        // matches, which would wrap a 32-bit counter and restart
        // idx at 0, overwriting the buffer.
        const unsigned long long idx = atomicAdd(out_count, 1ull);
        if (idx < MAX_MATCHES)
          out[idx] = {x, y, z};
        bi = -1;
      }
    }
  }
}

namespace {

using Count = cuda::MatchBuffer::Count;
using Kernel = void (*)(SearchBounds, int, Count, Count*, MatchResult*, Count*);

[[nodiscard]] Kernel kernel_for(Version version) {
  switch (version) {
  case Version::legacy:
    return match_formation_kernel<Version::legacy>;
  case Version::vanilla12:
    return match_formation_kernel<Version::vanilla12>;
  case Version::sodium:
    return match_formation_kernel<Version::sodium>;
  case Version::sodium19:
    return match_formation_kernel<Version::sodium19>;
  case Version::modern:
    break;
  }
  return match_formation_kernel<Version::modern>;
}

[[nodiscard]] DevBlock device_block(const BlockInfo& block) {
  return {.z_term = static_cast<long long>(block.z) * 116129781LL,
          .x_term = static_cast<unsigned int>(block.x) * 3129871u,
          .dy = block.y,
          .w = block.rotation | ((block.is_side ? 1 : 3) << 8)};
}

[[nodiscard]] Count checked_volume(const SearchBounds& bounds) {
  constexpr auto limit{std::numeric_limits<Count>::max()};
  const Count nx{bounds.nx()}, ny{bounds.ny()}, nz{bounds.nz()};
  if (ny > limit / nz || nx > limit / (ny * nz))
    throw std::invalid_argument{"search volume too large"};
  return nx * ny * nz;
}

// Persistent threads: enough blocks to fill the device once, or fewer when
// the search is too small to need them all.
[[nodiscard]] unsigned int grid_blocks(Kernel kernel, std::size_t shared_bytes, Count positions) {
  int device{0};
  cuda::check(cudaGetDevice(&device));
  cudaDeviceProp properties;
  cuda::check(cudaGetDeviceProperties(&properties, device));
  int blocks_per_sm{0};
  cuda::check(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, TF_BLOCK_THREADS, shared_bytes));

  const Count chunk_size{static_cast<Count>(WARP) * TF_CHUNK_PER_LANE};
  const Count chunks{positions / chunk_size + (positions % chunk_size != 0)}; // one in flight per warp
  const Count blocks_for_work{(chunks * WARP + TF_BLOCK_THREADS - 1) / TF_BLOCK_THREADS};
  const Count blocks_on_device{static_cast<Count>(properties.multiProcessorCount) *
                               static_cast<Count>(std::max(blocks_per_sm, 1))};
  return static_cast<unsigned int>(std::max<Count>(1, std::min(blocks_on_device, blocks_for_work)));
}

[[nodiscard]] DirectionResult standard_search(const SearchBounds& bounds, std::span<const BlockInfo> blocks,
                                              Version version, Count positions) {
  std::vector<DevBlock> device_blocks;
  for (const auto& block : blocks)
    device_blocks.push_back(device_block(block));
  const auto shared_bytes{device_blocks.size() * sizeof(DevBlock)};
  cuda::check(cudaMemcpyToSymbol(d_blocks, device_blocks.data(), shared_bytes));

  const auto kernel{kernel_for(version)};
  const auto grid{grid_blocks(kernel, shared_bytes, positions)};
  const cuda::MatchBuffer matches;
  cuda::DeviceArray<Count> cursor{1};
  cursor.zero();

  cuda::Timer timer;
  kernel<<<grid, TF_BLOCK_THREADS, shared_bytes>>>(bounds, static_cast<int>(device_blocks.size()), positions,
                                                   cursor.get(), matches.matches(), matches.count());
  cuda::check(cudaGetLastError());
  const auto seconds{timer.stop_seconds()};

  auto result{matches.collect()};
  result.kernel_seconds = seconds;
  return result;
}

void trace(const char* message) {
  if (cuda_bitplanes::trace_enabled())
    std::fprintf(stderr, "CUDA search: %s\n", message);
}

} // namespace

DirectionResult run_search(const SearchBounds& bounds, const std::vector<BlockInfo>& blocks,
                           Version version) {
  if (blocks.empty() || blocks.size() > 2 * MAX_FORMATION_BLOCKS)
    throw std::invalid_argument{"formation must have 1 to " + std::to_string(2 * MAX_FORMATION_BLOCKS) +
                                " blocks"};
  const auto positions{checked_volume(bounds)};

  // Modern searches use bitplanes when the formation filters well enough and
  // the field fits in memory.
  if (shared_search_enabled(version)) {
    if (bitplanes::eligible(blocks)) {
      if (auto output{cuda_bitplanes::search(bounds, std::span{&blocks, 1})}) {
        trace("bitplanes");
        auto& result{output->directions.front()};
        result.kernel_seconds = output->kernel_seconds;
        return std::move(result);
      }
    }
    trace("baseline fallback");
  }
  return standard_search(bounds, blocks, version, positions);
}

bool shared_search_enabled(Version version) {
  return version == Version::modern && cuda_bitplanes::enabled();
}

SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                     std::span<const std::vector<BlockInfo>> directions, Version version) {
  return bitplanes::search_sharing_field(
      bounds, directions, version,
      [&](std::span<const std::vector<BlockInfo>> shared) -> std::optional<SharedSearchResult> {
        auto output{cuda_bitplanes::search(bounds, shared)};
        if (!output)
          return std::nullopt;
        if (cuda_bitplanes::trace_enabled())
          std::fprintf(stderr, "CUDA search: bitplanes shared (%zu directions)\n", shared.size());
        return SharedSearchResult{std::move(output->directions), output->kernel_seconds};
      });
}
