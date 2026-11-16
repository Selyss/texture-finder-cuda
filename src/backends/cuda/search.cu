#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>

#include "backends/cuda/bitplanes.cuh"
#include "texture/texture_rules.hpp"

#define CUDA_CHECK(call)                                                                                     \
  do {                                                                                                       \
    cudaError_t err_ = (call);                                                                               \
    if (err_ != cudaSuccess) {                                                                               \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));       \
      std::exit(2);                                                                                          \
    }                                                                                                        \
  } while (0)

const char* search_backend_name() {
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
template <int VERSION>
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
      int v;
      if (VERSION == LEGACY_VERSION)
        v = legacy_from_coord_random(mix >> 16, 4);
      else if (VERSION == VANILLA12_VERSION)
        v = vanilla12_from_mix(mix, 4);
      else if (VERSION == SODIUM_VERSION)
        v = sodium_from_coord_random(mix >> 16, 4);
      else if (VERSION == SODIUM19_VERSION)
        v = sodium19_from_coord_random(mix >> 16, 4);
      else
        v = modern_from_coord_random(mix >> 16, 4);
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
using namespace cuda_bitplanes;

// Preserve the per-call count, output limit, and ordering for both CUDA paths.
DirectionResult collect_matches(const MatchResult* output, const Word* counter) {
  Word count = 0;
  CUDA_CHECK(cudaMemcpy(&count, counter, sizeof(count), cudaMemcpyDeviceToHost));

  std::vector<MatchResult> matches(std::min<Word>(count, MAX_MATCHES));
  if (!matches.empty())
    CUDA_CHECK(
        cudaMemcpy(matches.data(), output, matches.size() * sizeof(MatchResult), cudaMemcpyDeviceToHost));
  std::sort(matches.begin(), matches.end(), [](const MatchResult& a, const MatchResult& b) {
    return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z);
  });
  return {std::move(matches), count, count > MAX_MATCHES};
}

size_t plane_budget() {
  constexpr size_t default_budget = size_t{1} << 30;
  const char* setting = std::getenv("TF_CUDA_BITPLANES_BYTES");
  if (!setting)
    return default_budget;
  const std::string_view value(setting);
  size_t bytes = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), bytes);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size()
             ? std::min(bytes, default_budget)
             : 0;
}

bool eligible_for_bitplanes(const std::vector<BlockInfo>& blocks) {
  std::array<int, STRIDE> filter_bits{};
  for (const auto& block : blocks)
    filter_bits[residue(-static_cast<int64_t>(block.z))] += block.is_side ? 1 : 2;
  // A weak coset can send most of its candidates through scalar verification.
  return *std::min_element(filter_bits.begin(), filter_bits.end()) >= 4;
}

struct DirectionChecks {
  std::vector<Check> checks;
  std::vector<VerifyCheck> verify;
  std::array<int, STRIDE + 1> offsets{}, verify_offsets{};
  PlaneCheck* device_checks = nullptr;
  VerifyCheck* device_verify = nullptr;
  int *device_offsets = nullptr, *device_verify_offsets = nullptr;
  MatchResult* output = nullptr;
  Word* counter = nullptr;
  std::vector<PlaneCheck> slab_checks;
};

std::optional<std::vector<DirectionResult>>
try_bitplanes(const SearchBounds& bounds, const std::vector<std::vector<BlockInfo>>& formations,
              float& kernel_ms) {
  int min_x = formations.front().front().x, max_x = min_x;
  int min_z = formations.front().front().z, max_z = min_z;
  std::vector<int> y_offsets;
  for (const auto& blocks : formations)
    for (const auto& block : blocks) {
      min_x = std::min(min_x, block.x);
      max_x = std::max(max_x, block.x);
      min_z = std::min(min_z, block.z);
      max_z = std::max(max_z, block.z);
      y_offsets.push_back(block.y);
    }

  const int64_t nx = static_cast<int64_t>(bounds.x_max) - bounds.x_min + 1;
  const int64_t ny = static_cast<int64_t>(bounds.y_max) - bounds.y_min + 1;
  const int64_t nz = static_cast<int64_t>(bounds.z_max) - bounds.z_min + 1;
  const int64_t field_x = static_cast<int64_t>(bounds.x_min) + min_x;
  const int64_t width = nx + static_cast<int64_t>(max_x) - min_x;
  const int64_t halo = static_cast<int64_t>(max_z) - min_z;
  std::sort(y_offsets.begin(), y_offsets.end());
  y_offsets.erase(std::unique(y_offsets.begin(), y_offsets.end()), y_offsets.end());
  constexpr int int_min = std::numeric_limits<int>::min();
  constexpr int int_max = std::numeric_limits<int>::max();
  if (nx <= 0 || ny <= 0 || nz <= 0 || width > int_max || ny > int_max || nz > int_max || field_x < int_min ||
      field_x + width - 1 > int_max || static_cast<int64_t>(bounds.z_min) + min_z < int_min ||
      static_cast<int64_t>(bounds.z_max) + max_z > int_max ||
      static_cast<int64_t>(bounds.y_min) + y_offsets.front() < int_min ||
      static_cast<int64_t>(bounds.y_max) + y_offsets.back() > int_max)
    return std::nullopt;

  const size_t field_words = ceil_div(static_cast<int>(width), WORD_BITS) + 1;
  const int candidate_words = ceil_div(static_cast<int>(nx), WORD_BITS);
  const int segments = ceil_div(candidate_words, THREADS);
  constexpr size_t result_bytes = MAX_MATCHES * sizeof(MatchResult);
  size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  // Leave room for results and the small metadata buffers.
  const size_t reserve = formations.size() * result_bytes + (size_t{16} << 20);
  const size_t budget = std::min(plane_budget(), free_bytes > reserve ? free_bytes - reserve : 0);
  const size_t words_budget = budget / sizeof(Planes);
  if (static_cast<size_t>(ny) > words_budget / field_words)
    return std::nullopt;

  const auto plane_y = compact_levels(bounds, y_offsets);
  if (plane_y.size() > words_budget / field_words)
    return std::nullopt;
  const size_t words_per_z = plane_y.size() * field_words;
  const size_t hashed_capacity = words_budget / words_per_z;
  const int64_t max_rows = static_cast<int64_t>(hashed_capacity) * STRIDE - halo;
  const size_t grid_per_z = static_cast<size_t>(ny) * segments;
  const int64_t planned_rows =
      std::min<int64_t>({nz, DEFAULT_SLAB_ROWS, max_rows, static_cast<int64_t>(int_max / grid_per_z)});
  if (planned_rows <= 0)
    return std::nullopt;
  const int slab_rows = static_cast<int>(planned_rows);

  const size_t plane_words = words_per_z * ((slab_rows + halo + STRIDE - 1) / STRIDE);
  std::vector<DirectionChecks> directions(formations.size());
  for (size_t i = 0; i < formations.size(); ++i) {
    auto& direction = directions[i];
    const auto& blocks = formations[i];
    for (int coset = 0; coset < STRIDE; ++coset) {
      for (const auto& block : blocks) {
        if (residue(-static_cast<int64_t>(block.z)) == coset) {
          const auto level = std::lower_bound(plane_y.begin(), plane_y.end(), bounds.y_min + block.y);
          direction.checks.push_back({block.x - min_x, static_cast<int>(level - plane_y.begin()), block.z,
                                      block.rotation, block.is_side});
        } else
          direction.verify.push_back({static_cast<int64_t>(block.z) * 116129781LL,
                                      static_cast<uint32_t>(block.x) * 3129871u, block.y, block.rotation,
                                      block.is_side ? 1 : 3});
      }
      direction.offsets[coset + 1] = static_cast<int>(direction.checks.size());
      direction.verify_offsets[coset + 1] = static_cast<int>(direction.verify.size());
    }
  }

  Planes* planes = nullptr;
  const cudaError_t allocation = cudaMalloc(&planes, plane_words * sizeof(Planes));
  if (allocation == cudaErrorMemoryAllocation) {
    cudaGetLastError();
    return std::nullopt;
  }
  CUDA_CHECK(allocation);

  auto upload = [](const auto& values) {
    using Element = typename std::decay_t<decltype(values)>::value_type;
    Element* device = nullptr;
    if (!values.empty()) {
      const size_t bytes = values.size() * sizeof(Element);
      CUDA_CHECK(cudaMalloc(&device, bytes));
      CUDA_CHECK(cudaMemcpy(device, values.data(), bytes, cudaMemcpyHostToDevice));
    }
    return device;
  };
  auto* device_y = upload(plane_y);
  for (auto& direction : directions) {
    direction.slab_checks.resize(direction.checks.size());
    CUDA_CHECK(cudaMalloc(&direction.device_checks, direction.slab_checks.size() * sizeof(PlaneCheck)));
    direction.device_verify = upload(direction.verify);
    direction.device_offsets = upload(direction.offsets);
    direction.device_verify_offsets = upload(direction.verify_offsets);
    CUDA_CHECK(cudaMalloc(&direction.output, result_bytes));
    CUDA_CHECK(cudaMalloc(&direction.counter, sizeof(Word)));
    CUDA_CHECK(cudaMemset(direction.counter, 0, sizeof(Word)));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int64_t offset = 0; offset < nz;) {
    const int rows = static_cast<int>(std::min<int64_t>(slab_rows, nz - offset));
    SearchBounds slab = bounds;
    slab.z_min = static_cast<int>(bounds.z_min + offset);
    slab.z_max = slab.z_min + (rows - 1);
    const int64_t halo_start = static_cast<int64_t>(slab.z_min) + min_z;
    const int64_t halo_end = static_cast<int64_t>(slab.z_max) + max_z;
    const int64_t first_z = halo_start + residue(-halo_start);
    const int hashed_rows = first_z <= halo_end ? static_cast<int>((halo_end - first_z) / STRIDE + 1) : 0;
    const Layout layout{slab,
                        static_cast<int>(nx),
                        static_cast<int>(ny),
                        rows,
                        static_cast<int>(field_x),
                        static_cast<int>(width),
                        static_cast<int>(field_words),
                        candidate_words,
                        static_cast<int>(plane_y.size()),
                        hashed_rows,
                        first_z};
    const size_t words = words_per_z * hashed_rows;
    if (words) {
      fill_planes<true, true><<<(words + THREADS - 1) / THREADS, THREADS>>>(layout, device_y, words, planes);
      CUDA_CHECK(cudaGetLastError());
    }
    for (auto& direction : directions) {
      prepare_plane_checks(layout, direction.checks, direction.offsets.data(), direction.slab_checks);
      // Copies and kernels are ordered on the default stream.
      CUDA_CHECK(cudaMemcpy(direction.device_checks, direction.slab_checks.data(),
                            direction.slab_checks.size() * sizeof(PlaneCheck), cudaMemcpyHostToDevice));
      match_words<MatchMode::Search, true, true><<<grid_per_z * rows, THREADS>>>(
          layout, direction.device_checks, direction.device_offsets, planes, nullptr, nullptr, 0,
          direction.device_verify, direction.device_verify_offsets, direction.output, direction.counter,
          MAX_MATCHES);
      CUDA_CHECK(cudaGetLastError());
    }
    offset += rows;
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<DirectionResult> results(directions.size());
  for (size_t i = 0; i < directions.size(); ++i) {
    const auto& direction = directions[i];
    results[i] = collect_matches(direction.output, direction.counter);
    CUDA_CHECK(cudaFree(direction.counter));
    CUDA_CHECK(cudaFree(direction.output));
    CUDA_CHECK(cudaFree(direction.device_verify_offsets));
    CUDA_CHECK(cudaFree(direction.device_offsets));
    CUDA_CHECK(cudaFree(direction.device_verify));
    CUDA_CHECK(cudaFree(direction.device_checks));
  }
  CUDA_CHECK(cudaFree(device_y));
  CUDA_CHECK(cudaFree(planes));
  return results;
}
} // namespace

DirectionResult run_search(const SearchBounds& bounds, const std::vector<BlockInfo>& tops_and_bottoms,
                           const std::vector<BlockInfo>& sides, int version) {
  if (tops_and_bottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS) {
    std::fprintf(stderr, "Formation exceeds %d blocks per face type\n", MAX_FORMATION_BLOCKS);
    std::exit(1);
  }

  // Tops first: a mod-4 check rejects 75% of candidates, a side check only
  // 50%, so the most selective checks run earliest.
  std::vector<DevBlock> blocks;
  blocks.reserve(tops_and_bottoms.size() + sides.size());
  for (const auto& t : tops_and_bottoms)
    blocks.push_back(
        {(long long)t.z * 116129781LL, (unsigned int)t.x * 3129871u, t.y, t.rotation | (3 << 8)});
  for (const auto& s : sides)
    blocks.push_back(
        {(long long)s.z * 116129781LL, (unsigned int)s.x * 3129871u, s.y, s.rotation | (1 << 8)});
  if (blocks.empty()) {
    std::fprintf(stderr, "Empty formation\n");
    std::exit(1);
  }

  const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
  const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
  const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
  // Two-step overflow guard: ny*nz itself must not overflow before it is
  // used as a divisor (nx/ny/nz are each >= 1 by construction).
  if (ny > (~0ull) / nz || nx > (~0ull) / (ny * nz)) {
    std::fprintf(stderr, "Search volume too large\n");
    std::exit(1);
  }
  const unsigned long long total = nx * ny * nz;

  if (const char* enabled = std::getenv("TF_CUDA_BITPLANES"); enabled && std::string_view(enabled) == "1") {
    const bool trace = std::getenv("TF_CUDA_BITPLANES_TRACE") != nullptr;
    if (version == MODERN_VERSION) {
      std::vector<std::vector<BlockInfo>> formations{tops_and_bottoms};
      formations.front().insert(formations.front().end(), sides.begin(), sides.end());
      if (eligible_for_bitplanes(formations.front())) {
        float ms = 0;
        if (auto results = try_bitplanes(bounds, formations, ms)) {
          if (trace)
            std::fputs("CUDA search: bitplanes\n", stderr);
          results->front().kernel_ms = ms;
          return std::move(results->front());
        }
      }
    }
    if (trace)
      std::fputs("CUDA search: baseline fallback\n", stderr);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(d_blocks, blocks.data(), blocks.size() * sizeof(DevBlock)));

  MatchResult* d_out = nullptr;
  unsigned long long* d_count = nullptr;
  unsigned long long* d_cursor = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, MAX_MATCHES * sizeof(MatchResult)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_cursor, 0, sizeof(unsigned long long)));

  // Persistent-thread launch sized to fill the device exactly once.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  const size_t shared_bytes = blocks.size() * sizeof(DevBlock);

  // Compile-time version dispatch: resolve the kernel instantiation once.
  using KernelFn =
      void (*)(SearchBounds, int, unsigned long long, unsigned long long*, MatchResult*, unsigned long long*);
  KernelFn kernel;
  switch (version) {
  case LEGACY_VERSION:
    kernel = match_formation_kernel<LEGACY_VERSION>;
    break;
  case VANILLA12_VERSION:
    kernel = match_formation_kernel<VANILLA12_VERSION>;
    break;
  case SODIUM_VERSION:
    kernel = match_formation_kernel<SODIUM_VERSION>;
    break;
  case SODIUM19_VERSION:
    kernel = match_formation_kernel<SODIUM19_VERSION>;
    break;
  default:
    kernel = match_formation_kernel<MODERN_VERSION>;
    break;
  }

  int blocks_per_sm = 0;
  CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, TF_BLOCK_THREADS, shared_bytes));
  const unsigned long long chunk_size = (unsigned long long)WARP * TF_CHUNK_PER_LANE;
  const unsigned long long chunks_needed = total / chunk_size + (total % chunk_size != 0);
  const unsigned long long warps_needed = chunks_needed; // one chunk in flight per warp
  unsigned long long grid =
      (unsigned long long)props.multiProcessorCount * (blocks_per_sm > 0 ? blocks_per_sm : 1);
  const unsigned long long grid_for_work = (warps_needed * WARP + TF_BLOCK_THREADS - 1) / TF_BLOCK_THREADS;
  if (grid > grid_for_work)
    grid = grid_for_work; // tiny searches don't need the whole device
  if (grid == 0)
    grid = 1;

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventRecord(ev_start));

  kernel<<<(unsigned int)grid, TF_BLOCK_THREADS, shared_bytes>>>(bounds, (int)blocks.size(), total, d_cursor,
                                                                 d_out, d_count);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  auto result = collect_matches(d_out, d_count);
  result.kernel_ms = ms;
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_cursor));
  return result;
}

bool shared_search_enabled(int version) {
  const char* enabled = std::getenv("TF_CUDA_BITPLANES");
  return version == MODERN_VERSION && enabled && std::string_view(enabled) == "1";
}

SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                     const std::vector<SearchFormation>& formations, int version) {
  SharedSearchResult result;
  result.directions.resize(formations.size());
  std::vector<size_t> shared_indices;
  std::vector<std::vector<BlockInfo>> eligible;
  if (shared_search_enabled(version)) {
    for (size_t i = 0; i < formations.size(); ++i) {
      const auto& formation = formations[i];
      if (formation.tops.size() > MAX_FORMATION_BLOCKS || formation.sides.size() > MAX_FORMATION_BLOCKS)
        continue; // run_search reports the input error.
      auto blocks = formation.tops;
      blocks.insert(blocks.end(), formation.sides.begin(), formation.sides.end());
      if (eligible_for_bitplanes(blocks)) {
        shared_indices.push_back(i);
        eligible.push_back(std::move(blocks));
      }
    }
  }
  std::optional<std::vector<DirectionResult>> shared;
  if (eligible.size() > 1) {
    float ms = 0;
    shared = try_bitplanes(bounds, eligible, ms);
    if (shared) {
      result.kernel_seconds += ms / 1000.0;
      if (std::getenv("TF_CUDA_BITPLANES_TRACE"))
        std::fprintf(stderr, "CUDA search: bitplanes shared (%zu directions)\n", shared->size());
    }
  }
  size_t next_shared = 0;
  for (size_t i = 0; i < formations.size(); ++i) {
    if (shared && next_shared < shared_indices.size() && shared_indices[next_shared] == i) {
      result.directions[i] = std::move((*shared)[next_shared++]);
      continue;
    }
    result.directions[i] = run_search(bounds, formations[i].tops, formations[i].sides, version);
    result.kernel_seconds += result.directions[i].kernel_ms / 1000.0;
  }
  return result;
}
