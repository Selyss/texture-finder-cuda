// Prototype 3: x-axis bitplanes + z-row lattice, for searches with few y levels.
// Build: clang++ -O2 -std=c++17 flat.cpp -o flat
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <tuple>
#include <vector>
#include "texture/texture_rules.hpp"
struct Blk {
  int dx, dy, dz, expected, mask;
  uint32_t x_term;
  int64_t z_term;
};
struct Box {
  int x0, x1, y0, y1, z0, z1;
};
using Match = std::tuple<int, int, int>;
static inline int out_stage(int version, int64_t mix) {
  switch (version) {
  case 1:
    return legacy_from_coord_random(mix >> 16, 4);
  case 2:
    return vanilla12_from_mix(mix, 4);
  case 3:
    return sodium_from_coord_random(mix >> 16, 4);
  case 4:
    return sodium19_from_coord_random(mix >> 16, 4);
  default:
    return modern_from_coord_random(mix >> 16, 4);
  }
}
template <int V>
static std::vector<Match> scan_ref(const Box& b, const std::vector<Blk>& f, double* evals) {
  std::vector<Match> out;
  uint64_t n = 0;
  for (int z = b.z0; z <= b.z1; z++) {
    int64_t hz = (int64_t)z * 116129781LL;
    for (int x = b.x0; x <= b.x1; x++) {
      uint32_t hx = (uint32_t)x * 3129871u;
      for (int y = b.y0; y <= b.y1; y++) {
        size_t i = 0;
        for (; i < f.size(); i++) {
          n++;
          int v = out_stage(V, coord_mix_from_parts(hx + f[i].x_term, hz + f[i].z_term, y + f[i].dy));
          if ((v & f[i].mask) != f[i].expected)
            break;
        }
        if (i == f.size())
          out.emplace_back(x, y, z);
      }
    }
  }
  *evals = (double)n;
  return out;
}
static inline int pmod(long v, int s) {
  int r = (int)(v % s);
  return r < 0 ? r + s : r;
}

static int choose_stride(const std::vector<Blk>& f, int max_s) {
  int best = 1;
  double bc = 1e30;
  for (int s = 1; s <= max_s; s++) {
    std::vector<double> surv(s, 1.0);
    for (auto& k : f)
      surv[pmod(-(long)k.dz, s)] *= k.mask == 3 ? 0.25 : 0.5;
    double c = 3.0 / s;
    for (double v : surv)
      c += v * 1.34 * 3.0 / s;
    if (c < bc - 1e-12) {
      bc = c;
      best = s;
    }
  }
  return best;
}

// Only world rows with pz % s == 0 are hashed. Each hashed row is stored as two
// bit arrays along x, so one block check covers 64 candidate x at once.
template <int V>
static std::vector<Match> scan_flat(const Box& b, const std::vector<Blk>& f, int s, double* field_evals,
                                    double* full_evals) {
  std::vector<Match> out;
  uint64_t n_field = 0, n_full = 0;
  int dxmin = 1 << 30, dxmax = -(1 << 30), dzmin = dxmin, dzmax = dxmax;
  std::map<int, int> dy_idx;
  for (auto& k : f) {
    dxmin = std::min(dxmin, k.dx);
    dxmax = std::max(dxmax, k.dx);
    dzmin = std::min(dzmin, k.dz);
    dzmax = std::max(dzmax, k.dz);
    dy_idx.emplace(k.dy, 0);
  }
  int ndy = 0;
  for (auto& kv : dy_idx)
    kv.second = ndy++;
  const long nx = (long)b.x1 - b.x0 + 1, pxmin = (long)b.x0 + dxmin, width = nx + (dxmax - dxmin);
  const size_t words = (size_t)(width + 63) / 64 + 2;
  const int nrows = dzmax - dzmin + 1;
  std::vector<uint64_t> P0((size_t)ndy * nrows * words), P1(P0.size());
  std::vector<std::vector<int>> on(s), off(s);
  for (int r = 0; r < s; r++)
    for (int i = 0; i < (int)f.size(); i++)
      (pmod((long)f[i].dz + r, s) == 0 ? on : off)[r].push_back(i);

  for (long y = b.y0; y <= b.y1; y++) {
    auto fill_row = [&](long pz) {
      if (pmod(pz, s) != 0)
        return;
      const int64_t hz = (int64_t)pz * 116129781LL;
      for (auto& kv : dy_idx) {
        uint64_t *r0 = &P0[((size_t)kv.second * nrows + pmod(pz, nrows)) * words],
                 *r1 = &P1[((size_t)kv.second * nrows + pmod(pz, nrows)) * words];
        const int wy = (int)(y + kv.first);
        uint32_t hx = (uint32_t)pxmin * 3129871u;
        for (size_t w = 0; w * 64 < (size_t)width; w++) {
          uint64_t a0 = 0, a1 = 0;
          const int n = (int)std::min<long>(64, width - (long)w * 64);
          for (int j = 0; j < n; j++, hx += 3129871u) {
            const unsigned v = (unsigned)out_stage(V, coord_mix_from_parts(hx, hz, wy));
            a0 |= (uint64_t)(v & 1) << j;
            a1 |= (uint64_t)(v >> 1) << j;
          }
          r0[w] = a0;
          r1[w] = a1;
          n_field += n;
        }
      }
    };
    std::vector<const uint64_t*> row0(f.size()), row1(f.size());
    std::vector<int> off0(f.size());
    for (long z = b.z0; z <= b.z1; z++) {
      if (z == b.z0)
        for (long pz = z + dzmin; pz <= z + dzmax; pz++)
          fill_row(pz);
      else
        fill_row(z + dzmax);
      const int r = pmod(z, s);
      const int64_t hz = (int64_t)z * 116129781LL;
      for (int i : on[r]) {
        const size_t base = ((size_t)dy_idx[f[i].dy] * nrows + pmod(z + f[i].dz, nrows)) * words;
        row0[i] = &P0[base];
        row1[i] = &P1[base];
        off0[i] = f[i].dx - dxmin;
      }
      for (long w = 0; w * 64 < nx; w++) {
        const int n = (int)std::min<long>(64, nx - w * 64);
        uint64_t M = n == 64 ? ~0ull : ((1ull << n) - 1);
        for (int i : on[r]) {
          const int sh = off0[i]; // candidate bit j reads row bit 64w + j + sh
          const size_t q = (size_t)w + (size_t)(sh >> 6);
          const int rs = sh & 63;
          uint64_t q0 = row0[i][q] >> rs, q1 = row1[i][q] >> rs;
          if (rs) {
            q0 |= row0[i][q + 1] << (64 - rs);
            q1 |= row1[i][q + 1] << (64 - rs);
          }
          uint64_t eq = (f[i].expected & 1) ? q0 : ~q0;
          if (f[i].mask == 3)
            eq &= (f[i].expected & 2) ? q1 : ~q1;
          M &= eq;
          if (!M)
            break;
        }
        while (M) {
          const int j = __builtin_ctzll(M);
          M &= M - 1;
          const int x = (int)(b.x0 + w * 64 + j);
          const uint32_t hx = (uint32_t)x * 3129871u;
          bool ok = true;
          for (int i : off[r]) {
            n_full++;
            int v = out_stage(V, coord_mix_from_parts(hx + f[i].x_term, hz + f[i].z_term, (int)y + f[i].dy));
            if ((v & f[i].mask) != f[i].expected) {
              ok = false;
              break;
            }
          }
          if (ok)
            out.emplace_back(x, (int)y, (int)z);
        }
      }
    }
  }
  *field_evals = (double)n_field;
  *full_evals = (double)n_full;
  return out;
}
static std::vector<Blk> load_formation(const char* path) {
  std::vector<Blk> tops, sides;
  std::ifstream in(path);
  int x, y, z, r, s;
  while (in >> x >> y >> z >> r >> s) {
    Blk b{x, y, z, s ? r % 2 : r, s ? 1 : 3, (uint32_t)x * 3129871u, (int64_t)z * 116129781LL};
    (s ? sides : tops).push_back(b);
  }
  tops.insert(tops.end(), sides.begin(), sides.end());
  return tops;
}
template <int V>
static bool compare(const char* name, const Box& b, const std::vector<Blk>& f, int s, bool timing) {
  using clk = std::chrono::steady_clock;
  double e1, fe, fu;
  auto t0 = clk::now();
  auto r = scan_ref<V>(b, f, &e1);
  auto t1 = clk::now();
  auto q = scan_flat<V>(b, f, s, &fe, &fu);
  auto t2 = clk::now();
  std::sort(r.begin(), r.end());
  std::sort(q.begin(), q.end());
  bool ok = r == q;
  double cand = (double)(b.x1 - b.x0 + 1) * (b.y1 - b.y0 + 1) * (b.z1 - b.z0 + 1);
  double tr = std::chrono::duration<double>(t1 - t0).count(),
         ts = std::chrono::duration<double>(t2 - t1).count();
  if (timing || !ok)
    std::printf(
        "%-22s v%d s=%-2d %s matches=%zu | ref %.3fs %.2f full/cand | new %.3fs %.3f field + %.4f full /cand | %.1fx (%.0f M cand/s)\n",
        name, V, s, ok ? "IDENTICAL" : "MISMATCH", r.size(), tr, e1 / cand, ts, fe / cand, fu / cand, tr / ts,
        cand / ts / 1e6);
  return ok;
}
int main(int argc, char** argv) {
  auto f = load_formation(argc > 1 ? argv[1] : "formation_a.txt");
  bool ok = true;
  std::mt19937_64 rng(4242);
  auto ri = [&](long lo, long hi) {
    return (int)(lo + (long)(rng() % (uint64_t)(hi - lo + 1)));
  };
  int trials = 0;
  for (int it = 0; it < 600; it++) {
    std::vector<Blk> g;
    int nb = ri(1, 7);
    for (int i = 0; i < nb; i++) {
      int side = ri(0, 2) == 0, x = ri(-70, 70), y = ri(-6, 6), z = ri(-5, 5), r = side ? ri(0, 1) : ri(0, 3);
      g.push_back({x, y, z, r, side ? 1 : 3, (uint32_t)x * 3129871u, (int64_t)z * 116129781LL});
    }
    int far = ri(0, 3) == 0;
    int cx = far ? ri(-29999000, 29999000) : ri(-200000, 200000),
        cz = far ? ri(-29999000, 29999000) : ri(-200000, 200000);
    int ya = ri(-70, 320);
    Box b{cx, cx + ri(0, 300), ya, ya + ri(0, 2), cz, cz + ri(0, 20)};
    int s = ri(1, 9);
    ok &= compare<0>("random", b, g, s, false);
    ok &= compare<1>("random", b, g, s, false);
    ok &= compare<2>("random", b, g, s, false);
    ok &= compare<3>("random", b, g, s, false);
    ok &= compare<4>("random", b, g, s, false);
    trials += 5;
  }
  std::printf("randomised equivalence: %d trials -> %s\n", trials, ok ? "all identical" : "MISMATCH");
  Box flat{-116000, -101000, -54, -54, -74000, -66000};
  Box few{-113000, -104000, -56, -52, -72000, -67000};
  int s = choose_stride(f, 16);
  std::printf("chosen z stride: %d\n", s);
  ok &= compare<0>("single level, s=1", flat, f, 1, true);
  for (int t : {2, 3, 6})
    ok &= compare<0>("single level", flat, f, t, true);
  ok &= compare<1>("single level", flat, f, s, true);
  ok &= compare<0>("5 levels", few, f, s, true);
  return ok ? 0 : 1;
}
