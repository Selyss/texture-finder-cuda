// Prototype: y-column forward-difference sweep vs the reference per-candidate scan.
// Build: see docs/RNG_REDUCTION_LOG.md
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>
#include "texture/texture_rules.hpp"

struct Blk { int dx, dy, dz, expected, mask; uint32_t xTerm; int64_t zTerm; };
struct Box { int x0, x1, y0, y1, z0, z1; };
using Match = std::tuple<int, int, int>;

static const uint64_t C = 42317861ull;

static inline int outStage(int version, int64_t mix)
{
    switch (version)
    {
    case 1: return legacyFromCoordRandom(mix >> 16, 4);
    case 2: return vanilla12FromMix(mix, 4);
    case 3: return sodiumFromCoordRandom(mix >> 16, 4);
    case 4: return sodium19FromCoordRandom(mix >> 16, 4);
    default: return modernFromCoordRandom(mix >> 16, 4);
    }
}

// Same per-evaluation work as the CUDA kernel: precomputed x/z terms, early exit.
template <int V>
static std::vector<Match> scanRef(const Box &b, const std::vector<Blk> &f, uint64_t *evals)
{
    std::vector<Match> out; uint64_t n = 0;
    for (int z = b.z0; z <= b.z1; z++)
    {
        int64_t hz = (int64_t)z * 116129781LL;
        for (int x = b.x0; x <= b.x1; x++)
        {
            uint32_t hx = (uint32_t)x * 3129871u;
            for (int y = b.y0; y <= b.y1; y++)
            {
                size_t i = 0;
                for (; i < f.size(); i++)
                {
                    n++;
                    int v = outStage(V, coordMixFromParts(hx + f[i].xTerm, hz + f[i].zTerm, y + f[i].dy));
                    if ((v & f[i].mask) != f[i].expected) break;
                }
                if (i == f.size()) out.emplace_back(x, y, z);
            }
        }
    }
    *evals = n;
    return out;
}

// Block 0 is swept along the whole y column with second-order forward
// differences (no multiplies for the mix); survivors fall back to the
// reference evaluation for blocks 1..n-1.
template <int V>
static std::vector<Match> scanSweep(const Box &b, const std::vector<Blk> &f, uint64_t *fullEvals)
{
    std::vector<Match> out; uint64_t n = 0;
    const Blk &b0 = f[0];
    for (int z = b.z0; z <= b.z1; z++)
    {
        int64_t hz = (int64_t)z * 116129781LL;
        for (int x = b.x0; x <= b.x1; x++)
        {
            uint32_t hx = (uint32_t)x * 3129871u;
            const uint64_t K = (uint64_t)((int64_t)(int32_t)(hx + b0.xTerm) ^ (hz + b0.zTerm));
            // world-y range of block 0, split at the sign: for yy < 0,
            // K ^ sext(yy) == ~K ^ (~yy) with ~yy >= 0.
            for (int neg = 0; neg < 2; neg++)
            {
                long lo = (long)b.y0 + b0.dy, hi = (long)b.y1 + b0.dy;
                uint64_t Kx; long a, e; // segment range [a, e] in the non-negative representation
                if (neg) { if (lo > -1) continue; long h2 = hi < -1 ? hi : -1; a = ~h2; e = ~lo; Kx = ~K; }
                else     { if (hi < 0) continue;  a = lo > 0 ? lo : 0;        e = hi;  Kx = K; }
                // decompose [a, e] into aligned power-of-two segments
                while (a <= e)
                {
                    int k = 0;
                    while (k < 30 && (a & ((2L << k) - 1)) == 0 && a + (2L << k) - 1 <= e) k++;
                    const uint64_t m = (1ull << k) - 1;
                    const uint64_t ub = (Kx ^ (uint64_t)a) & ~m; // segment base of the hash input
                    const uint64_t kl = Kx & m;                  // permutation of y inside the segment
                    uint64_t l = ub * ub * C + ub * 11ull;       // P(ub)
                    uint64_t d = C * (2 * ub + 1) + 11ull;       // P(ub+1) - P(ub)
                    for (uint64_t t = 0; t <= m; t++)
                    {
                        int v = outStage(V, (int64_t)l);
                        if ((v & b0.mask) == b0.expected)
                        {
                            long yrep = a + (long)(t ^ kl);
                            int y = (int)((neg ? ~yrep : yrep) - b0.dy);
                            size_t i = 1;
                            for (; i < f.size(); i++)
                            {
                                n++;
                                int w = outStage(V, coordMixFromParts(hx + f[i].xTerm, hz + f[i].zTerm, y + f[i].dy));
                                if ((w & f[i].mask) != f[i].expected) break;
                            }
                            if (i == f.size()) out.emplace_back(x, y, z);
                        }
                        l += d;
                        d += 2 * C;
                    }
                    a += (long)m + 1;
                }
            }
        }
    }
    *fullEvals = n;
    return out;
}

static std::vector<Blk> loadFormation(const char *path)
{
    std::vector<Blk> tops, sides; std::ifstream in(path); int x, y, z, r, s;
    while (in >> x >> y >> z >> r >> s)
    {
        Blk b{x, y, z, s ? r % 2 : r, s ? 1 : 3, (uint32_t)x * 3129871u, (int64_t)z * 116129781LL};
        (s ? sides : tops).push_back(b);
    }
    tops.insert(tops.end(), sides.begin(), sides.end());
    return tops;
}

template <int V>
static bool compare(const char *name, const Box &b, const std::vector<Blk> &f, bool timing)
{
    using clk = std::chrono::steady_clock;
    uint64_t e1, e2;
    auto t0 = clk::now(); auto r = scanRef<V>(b, f, &e1);
    auto t1 = clk::now(); auto s = scanSweep<V>(b, f, &e2);
    auto t2 = clk::now();
    std::sort(r.begin(), r.end()); std::sort(s.begin(), s.end());
    bool ok = r == s;
    double cand = (double)(b.x1 - b.x0 + 1) * (b.y1 - b.y0 + 1) * (b.z1 - b.z0 + 1);
    if (timing || !ok)
        std::printf("%-28s v%d %s matches=%zu  ref %.3fs (%.2f full evals/cand)  sweep %.3fs (%.2f full evals/cand)  speedup %.2fx\n",
                    name, V, ok ? "IDENTICAL" : "MISMATCH", r.size(),
                    std::chrono::duration<double>(t1 - t0).count(), e1 / cand,
                    std::chrono::duration<double>(t2 - t1).count(), e2 / cand,
                    std::chrono::duration<double>(t1 - t0).count() / std::chrono::duration<double>(t2 - t1).count());
    if (timing) for (auto &m : r) if (r.size() <= 3) std::printf("    match (%d, %d, %d)\n", std::get<0>(m), std::get<1>(m), std::get<2>(m));
    return ok;
}

int main(int argc, char **argv)
{
    const char *fix = argc > 1 ? argv[1] : "formation_a.txt";
    auto f = loadFormation(fix);
    bool ok = true;

    // 1. Randomised equivalence: weak formations (many matches), sides, negative
    //    and sign-straddling y ranges, world-border coordinates, all versions.
    std::mt19937_64 rng(12345);
    auto ri = [&](long lo, long hi) { return (int)(lo + (long)(rng() % (uint64_t)(hi - lo + 1))); };
    int trials = 0;
    for (int it = 0; it < 400; it++)
    {
        std::vector<Blk> g; int nb = ri(1, 4);
        for (int i = 0; i < nb; i++)
        {
            int side = ri(0, 2) == 0, x = ri(-6, 6), y = ri(-40, 40), z = ri(-6, 6), r = side ? ri(0, 1) : ri(0, 3);
            g.push_back({x, y, z, r, side ? 1 : 3, (uint32_t)x * 3129871u, (int64_t)z * 116129781LL});
        }
        int far = ri(0, 3) == 0;
        int cx = far ? ri(-29999900, 29999900) : ri(-200000, 200000), cz = far ? ri(-29999900, 29999900) : ri(-200000, 200000);
        int ya = ri(-2100, 2100), yb = ya + ri(0, 700);
        if (ri(0, 2) == 0) { ya = ri(-64, 319); yb = ri(ya, 319); }
        Box b{cx, cx + ri(0, 12), ya, yb, cz, cz + ri(0, 12)};
        ok &= compare<0>("random", b, g, false); ok &= compare<1>("random", b, g, false);
        ok &= compare<2>("random", b, g, false); ok &= compare<3>("random", b, g, false);
        ok &= compare<4>("random", b, g, false);
        trials += 5;
    }
    std::printf("randomised equivalence: %d box/formation/version trials -> %s\n", trials, ok ? "all identical" : "MISMATCH");

    // 2. Timing on slices of the reference workload around the known match.
    Box w{-109200, -108200, -64, -40, -70000, -69500};      // W's y range (25 tall)
    Box full{-108900, -108600, -64, 319, -69900, -69600};   // full build height (384 tall)
    Box flat{-111000, -106000, -54, -54, -71000, -68500};   // single y level
    ok &= compare<0>("W slice, y[-64,-40]", w, f, true);
    ok &= compare<1>("W slice, y[-64,-40]", w, f, true);
    ok &= compare<0>("full height, y[-64,319]", full, f, true);
    ok &= compare<0>("single level, y=-54", flat, f, true);
    return ok ? 0 : 1;
}
