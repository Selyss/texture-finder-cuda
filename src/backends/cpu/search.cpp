// Portable multithreaded CPU backend: same contract as the CUDA kernel,
// built from the same src/texture/texture_rules.hpp. Threads pull chunks of the
// linearized volume from an atomic cursor (the CPU analog of the GPU's
// chunk-cursor design) and walk them with the same incremental
// decompose-once-then-carry scheme.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "search/backend.hpp"
#include "texture/texture_rules.hpp"

const char* search_backend_name() {
  return "CPU";
}

namespace {

constexpr unsigned long long CHUNK = 1u << 16;

struct Shared {
  SearchBounds b;
  const BlockInfo* blocks;
  int n_blocks;
  int version;
  unsigned long long total;
  long long nx, nz;
  std::atomic<unsigned long long> cursor{0};
  std::atomic<unsigned long long> found{0};
  std::mutex sink;
  std::vector<MatchResult> results;
};

void worker(Shared& S) {
  std::vector<MatchResult> local;
  unsigned long long my_found = 0;

  for (;;) {
    const unsigned long long base = S.cursor.fetch_add(CHUNK);
    if (base >= S.total)
      break;
    const unsigned long long end = std::min(base + CHUNK, S.total);

    unsigned long long r = base;
    int x = S.b.x_min + (int)(r % (unsigned long long)S.nx);
    r /= (unsigned long long)S.nx;
    int z = S.b.z_min + (int)(r % (unsigned long long)S.nz);
    r /= (unsigned long long)S.nz;
    int y = S.b.y_min + (int)r;

    for (unsigned long long g = base; g < end; g++) {
      bool ok = true;
      for (int i = 0; i < S.n_blocks; i++) {
        const BlockInfo& bi = S.blocks[i];
        const int mod = bi.is_side ? MOD_SIDE : MOD_TOP_BOTTOM;
        const int v = get_texture_for_version(S.version, x + bi.x, y + bi.y, z + bi.z, mod);
        if (v != bi.rotation) {
          ok = false;
          break;
        }
      }
      if (ok) {
        my_found++;
        if (local.size() < MAX_MATCHES)
          local.push_back({x, y, z});
      }

      if (++x > S.b.x_max) {
        x = S.b.x_min;
        if (++z > S.b.z_max) {
          z = S.b.z_min;
          y++;
        }
      }
    }
  }

  S.found.fetch_add(my_found);
  std::lock_guard<std::mutex> lock(S.sink);
  for (const auto& m : local) {
    if (S.results.size() >= MAX_MATCHES)
      break;
    S.results.push_back(m);
  }
}

} // namespace

DirectionResult run_search(const SearchBounds& bounds, const std::vector<BlockInfo>& tops_and_bottoms,
                           const std::vector<BlockInfo>& sides, int version) {
  if (tops_and_bottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS) {
    std::fprintf(stderr, "Formation exceeds %d blocks per face type\n", MAX_FORMATION_BLOCKS);
    std::exit(1);
  }

  // Tops first: a mod-4 check rejects 75% of candidates, a side check 50%.
  std::vector<BlockInfo> blocks = tops_and_bottoms;
  blocks.insert(blocks.end(), sides.begin(), sides.end());
  if (blocks.empty()) {
    std::fprintf(stderr, "Empty formation\n");
    std::exit(1);
  }

  Shared S;
  S.b = bounds;
  S.blocks = blocks.data();
  S.n_blocks = (int)blocks.size();
  S.version = version;
  S.nx = (long long)bounds.x_max - bounds.x_min + 1;
  S.nz = (long long)bounds.z_max - bounds.z_min + 1;
  const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
  const unsigned long long nz_u = (unsigned long long)S.nz;
  const unsigned long long nx_u = (unsigned long long)S.nx;
  if (ny > (~0ull) / nz_u || nx_u > (~0ull) / (ny * nz_u)) {
    std::fprintf(stderr, "Search volume too large\n");
    std::exit(1);
  }
  S.total = nx_u * ny * nz_u;

  const auto t0 = std::chrono::steady_clock::now();

  unsigned n = std::thread::hardware_concurrency();
  if (n == 0)
    n = 4;
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < n; i++)
    pool.emplace_back(worker, std::ref(S));
  for (auto& t : pool)
    t.join();

  const auto t1 = std::chrono::steady_clock::now();
  const auto ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

  const unsigned long long count = S.found.load();

  std::sort(S.results.begin(), S.results.end(), [](const MatchResult& a, const MatchResult& b) {
    return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z);
  });
  return {std::move(S.results), count, count > MAX_MATCHES, ms};
}

bool shared_search_enabled(int) {
  return false;
}

SharedSearchResult run_shared_search(const SearchBounds& bounds,
                                     const std::vector<SearchFormation>& formations, int version) {
  SharedSearchResult result;
  result.directions.resize(formations.size());
  for (size_t i = 0; i < formations.size(); ++i) {
    result.directions[i] = run_search(bounds, formations[i].tops, formations[i].sides, version);
    result.kernel_seconds += result.directions[i].kernel_ms / 1000.0;
  }
  return result;
}
