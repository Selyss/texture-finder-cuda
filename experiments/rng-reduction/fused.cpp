// CPU oracle for the fused row design: fill and filter in one pass, no stored planes.
//
// One "thread" owns one hashed z row and one 64-position stretch of x. It
// hashes every world-y level of that stretch, then checks the candidates
// assigned to that row with shifts and ANDs, and verifies survivors against
// the blocks outside the row.
//
//   - Each candidate z uses the single hashed row holding most of its blocks.
//     A residue with no usable row is verified in full.
//   - A thread hashes 64 positions but owns 64 - (dx span) candidates, so it
//     never needs a neighbour's bits.
//
// Build and run from this directory:
//   clang++ -O2 -std=c++17 -I../../src fused.cpp -o fused
//   ./fused ../../test/fixtures/formation_a.txt
//
// Output: a randomised comparison against brute force (planted matches,
// strides 1-11), then hash and per-residue survivor counts for strides 5 and
// 6 on a slice of the reference workload. Modern version, top faces only.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "texture/texture_rules.hpp"
struct B {
  int dx, dy, dz, rot;
};
static int tex(int64_t x, int y, int64_t z) {
  return get_texture_for_version(0, (int)x, y, (int)z, 4);
}
static int pmod(int64_t a, int m) {
  int r = (int)(a % m);
  return r < 0 ? r + m : r;
}
struct Res {
  uint64_t survivors = 0, matches = 0, hashes = 0, verify_hashes = 0;
  std::vector<uint64_t> per_coset;
};
static Res fused(const std::vector<B>& f, int S, int64_t x0, int64_t x1, int y0, int y1, int64_t z0,
                 int64_t z1) {
  int min_dx = 1 << 30, max_dx = -(1 << 30), min_dy = 1 << 30, max_dy = -(1 << 30);
  for (auto& b : f) {
    min_dx = std::min(min_dx, b.dx);
    max_dx = std::max(max_dx, b.dx);
    min_dy = std::min(min_dy, b.dy);
    max_dy = std::max(max_dy, b.dy);
  }
  const int h = max_dx - min_dx, W = 64 - h;
  Res r;
  r.per_coset.assign(S, 0);
  if (W <= 0) {
    r.survivors = ~0ull;
    return r;
  }
  // assign each candidate residue to the dz row with most blocks
  std::vector<int> pick(S, INT32_MIN), cnt(S, 0);
  std::vector<int> dzs;
  for (auto& b : f)
    dzs.push_back(b.dz);
  std::sort(dzs.begin(), dzs.end());
  dzs.erase(std::unique(dzs.begin(), dzs.end()), dzs.end());
  for (int dz : dzs) {
    int n = 0;
    for (auto& b : f)
      n += b.dz == dz;
    int c = pmod(-dz, S);
    if (n > cnt[c]) {
      cnt[c] = n;
      pick[c] = dz;
    }
  }
  const int levels = (y1 - y0 + 1) + (max_dy - min_dy);
  std::vector<uint64_t> lo(levels), hi(levels);
  for (int c = 0; c < S; ++c)
    if (pick[c] == INT32_MIN) {
      // residue with no row: every candidate is a survivor
      for (int64_t z = z0; z <= z1; ++z)
        if (pmod(z, S) == c)
          for (int y = y0; y <= y1; ++y)
            for (int64_t x = x0; x <= x1; ++x) {
              ++r.survivors;
              ++r.per_coset[c];
              bool ok = true;
              for (auto& b : f) {
                ++r.verify_hashes;
                if (tex(x + b.dx, y + b.dy, z + b.dz) != b.rot) {
                  ok = false;
                  break;
                }
              }
              r.matches += ok;
            }
    }
  int64_t min_dz = dzs.front(), max_dz = dzs.back();
  int64_t zh0 = z0 + min_dz;
  zh0 += pmod(-zh0, S);
  for (int64_t zh = zh0; zh <= z1 + max_dz; zh += S)
    for (int64_t X = x0; X <= x1; X += W) {
      const int cand = (int)std::min<int64_t>(W, x1 - X + 1);
      for (int L = 0; L < levels; ++L) {
        uint64_t l = 0, hh = 0;
        for (int b = 0; b < cand + h; ++b) {
          int t = tex(X + min_dx + b, y0 + min_dy + L, zh);
          l |= (uint64_t)(t & 1) << b;
          hh |= (uint64_t)(t >> 1) << b;
          ++r.hashes;
        }
        lo[L] = l;
        hi[L] = hh;
      }
      for (int c = 0; c < S; ++c) {
        if (pick[c] == INT32_MIN)
          continue;
        const int64_t z = zh - pick[c];
        if (z < z0 || z > z1)
          continue;
        for (int y = y0; y <= y1; ++y) {
          uint64_t m = cand == 64 ? ~0ull : (1ull << cand) - 1;
          for (auto& b : f)
            if (b.dz == pick[c]) {
              const int L = (y - y0) + (b.dy - min_dy), s = b.dx - min_dx;
              const uint64_t l = lo[L] >> s, hh = hi[L] >> s;
              m &= (b.rot & 1) ? l : ~l;
              m &= (b.rot & 2) ? hh : ~hh;
            }
          for (; m; m &= m - 1) {
            const int64_t x = X + __builtin_ctzll(m);
            ++r.survivors;
            ++r.per_coset[pmod(z, S)];
            bool ok = true;
            for (auto& b : f)
              if (b.dz != pick[c]) {
                ++r.verify_hashes;
                if (tex(x + b.dx, y + b.dy, z + b.dz) != b.rot) {
                  ok = false;
                  break;
                }
              }
            r.matches += ok;
          }
        }
      }
    }
  return r;
}
static uint64_t brute(const std::vector<B>& f, int64_t x0, int64_t x1, int y0, int y1, int64_t z0,
                      int64_t z1) {
  uint64_t n = 0;
  for (int64_t z = z0; z <= z1; ++z)
    for (int y = y0; y <= y1; ++y)
      for (int64_t x = x0; x <= x1; ++x) {
        bool ok = true;
        for (auto& b : f)
          if (tex(x + b.dx, y + b.dy, z + b.dz) != b.rot) {
            ok = false;
            break;
          }
        n += ok;
      }
  return n;
}
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
    return 2;
  }
  std::vector<B> fix;
  {
    FILE* fp = fopen(argv[1], "r");
    if (!fp) {
      perror(argv[1]);
      return 2;
    }
    int a, b, c, d, e;
    while (fscanf(fp, "%d %d %d %d %d", &a, &b, &c, &d, &e) == 5)
      fix.push_back({a, b, c, d});
    fclose(fp);
  }
  std::mt19937_64 rng(7);
  int bad = 0, trials = 0;
  for (int t = 0; t < 400; ++t) {
    // random geometry, rotations planted from a real origin so >=1 match exists
    int n = 1 + rng() % 10;
    std::vector<B> f;
    int64_t ox = (int64_t)(rng() % 400000) - 200000, oz = (int64_t)(rng() % 400000) - 200000;
    int oy = (int)(rng() % 120) - 64;
    for (int i = 0; i < n; ++i) {
      B b{(int)(rng() % 9) - 4, (int)(rng() % 5) - 2, (int)(rng() % 9) - 4, 0};
      bool dup = false;
      for (auto& q : f)
        dup |= q.dx == b.dx && q.dy == b.dy && q.dz == b.dz;
      if (dup)
        continue;
      b.rot = tex(ox + b.dx, oy + b.dy, oz + b.dz);
      f.push_back(b);
    }
    int64_t x0 = ox - rng() % 150, x1 = ox + rng() % 150, z0 = oz - rng() % 20, z1 = oz + rng() % 20;
    int y0 = oy - rng() % 4, y1 = oy + rng() % 4;
    for (int S : {1, 2, 3, 5, 6, 7, 11}) {
      ++trials;
      auto r = fused(f, S, x0, x1, y0, y1, z0, z1);
      uint64_t e = brute(f, x0, x1, y0, y1, z0, z1);
      if (r.matches != e || e == 0) {
        ++bad;
        printf("MISMATCH t=%d S=%d got %llu want %llu\n", t, S, (unsigned long long)r.matches,
               (unsigned long long)e);
      }
    }
  }
  printf("random trials %d, mismatches %d\n", trials, bad);
  // W sub-slab: full y range, 120 z rows, 20000 x
  for (int S : {5, 6}) {
    int64_t x0 = -175000, x1 = -155001, z0 = -75000, z1 = -74881;
    int y0 = -64, y1 = -40;
    auto r = fused(fix, S, x0, x1, y0, y1, z0, z1);
    double cands = 20000.0 * 25 * 120;
    printf(
        "stride %d: candidates %.0f hashes/cand %.4f survivors %llu (%.4f%%) verifyHashes/survivor %.3f matches %llu  per coset:",
        S, cands, r.hashes / cands, (unsigned long long)r.survivors, 100.0 * r.survivors / cands,
        (double)r.verify_hashes / r.survivors, (unsigned long long)r.matches);
    for (auto v : r.per_coset)
      printf(" %llu", (unsigned long long)v);
    printf("\n");
  }
  return bad != 0;
}
