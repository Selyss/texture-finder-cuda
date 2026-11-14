// Prototype 2: hash-field bitplanes + sparse sublattice, vs the reference scan.
// Build: clang++ -O2 -std=c++17 lattice.cpp -o lattice
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>
#include "texture.cuh"

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

template <int V>
static std::vector<Match> scanRef(const Box &b, const std::vector<Blk> &f, double *evals)
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
    *evals = (double)n;
    return out;
}

static inline int pmod(long v, int s) { int r = (int)(v % s); return r < 0 ? r + s : r; }

struct Lattice { int s, a, b; double cost; };

// Pick the sublattice {(px,pz): a*px + b*pz == 0 mod s} minimising modelled
// multiplies per candidate: sweepCost/s for the field, plus, for each
// candidate coset, the chance of surviving the blocks that land on the
// lattice times ~1.33 full evaluations of the rest.
static Lattice chooseLattice(const std::vector<Blk> &f, double sweepCost, int maxS)
{
    Lattice best{1, 1, 0, 1e30};
    for (int s = 1; s <= maxS; s++)
        for (int a = 0; a < s || (s == 1 && a == 0); a++)
        {
            for (int b = 0; b < s || (s == 1 && b == 0); b++)
            {
                if (s > 1 && std::gcd(std::gcd(a, b), s) != 1) continue;
                std::vector<double> surv(s, 1.0);
                for (auto &k : f)
                    surv[pmod(-(long)a * k.dx - (long)b * k.dz, s)] *= k.mask == 3 ? 0.25 : 0.5;
                double c = sweepCost / s;
                for (int r = 0; r < s; r++) c += surv[r] * 1.34 * 3.0 / s;
                if (c < best.cost - 1e-12) best = {s, s == 1 ? 1 : a, s == 1 ? 0 : b, c};
            }
            if (s == 1) break;
        }
    return best;
}

// Fill the two bitplanes of one world column: bit i <-> world y = wlo + i.
template <int V>
static inline int fillColumn(int px, int pz, long wlo, long whi, uint64_t need, uint64_t &p0, uint64_t &p1)
{
    const uint64_t K = (uint64_t)((int64_t)(int32_t)((uint32_t)px * 3129871u) ^ ((int64_t)pz * 116129781LL));
    uint64_t b0 = 0, b1 = 0; int evals = 0;
    for (int neg = 0; neg < 2; neg++)
    {
        uint64_t Kx; long a, e;
        if (neg) { if (wlo > -1) continue; long h2 = whi < -1 ? whi : -1; a = ~h2; e = ~wlo; Kx = ~K; }
        else     { if (whi < 0) continue;  a = wlo > 0 ? wlo : 0;         e = whi;  Kx = K; }
        while (a <= e)
        {
            int k = 0;
            while (k < 6 && (a & ((2L << k) - 1)) == 0 && a + (2L << k) - 1 <= e) k++;
            const uint64_t m = (1ull << k) - 1;
            // skip segments none of whose world-y levels any block ever reads
            const long lowIdx = (neg ? ~(a + (long)m) : a) - wlo;
            const uint64_t segMask = (m == ~0ull >> 0 ? ~0ull : (k == 6 ? ~0ull : ((1ull << (m + 1)) - 1))) << lowIdx;
            if (!(segMask & need)) { a += (long)m + 1; continue; }
            evals += (int)m + 1;
            const uint64_t ub = (Kx ^ (uint64_t)a) & ~m, kl = Kx & m;
            uint64_t l = ub * ub * C + ub * 11ull;
            uint64_t d = k ? C * (2 * ub + 1) + 11ull : 0;
            for (uint64_t t = 0; t <= m; t++)
            {
                const unsigned v = (unsigned)outStage(V, (int64_t)l);
                const long yrep = a + (long)(t ^ kl);
                const int idx = (int)((neg ? ~yrep : yrep) - wlo);
                b0 |= (uint64_t)(v & 1) << idx;
                b1 |= (uint64_t)(v >> 1) << idx;
                l += d; d += 2 * C;
            }
            a += (long)m + 1;
        }
    }
    p0 = b0; p1 = b1;
    return evals;
}

template <int V>
static std::vector<Match> scanLattice(const Box &b, const std::vector<Blk> &f, Lattice L, double *fieldEvals, double *fullEvals)
{
    std::vector<Match> out; uint64_t nField = 0, nFull = 0;
    int dxmin = 1 << 30, dxmax = -(1 << 30), dymin = dxmin, dymax = dxmax, dzmin = dxmin, dzmax = dxmax;
    for (auto &k : f)
    {
        dxmin = std::min(dxmin, k.dx); dxmax = std::max(dxmax, k.dx);
        dymin = std::min(dymin, k.dy); dymax = std::max(dymax, k.dy);
        dzmin = std::min(dzmin, k.dz); dzmax = std::max(dzmax, k.dz);
    }
    const int dyext = dymax - dymin, slabH = 64 - dyext;
    if (slabH < 1) { std::fprintf(stderr, "formation taller than 64\n"); std::exit(1); }
    // Per candidate coset: blocks that land on the lattice (looked up), and the rest (evaluated only for survivors).
    std::vector<std::vector<int>> on(L.s), off(L.s);
    for (int r = 0; r < L.s; r++)
        for (int i = 0; i < (int)f.size(); i++)
            (pmod((long)L.a * f[i].dx + (long)L.b * f[i].dz + r, L.s) == 0 ? on : off)[r].push_back(i);

    const long pxmin = (long)b.x0 + dxmin, width = (long)b.x1 - b.x0 + 1 + (dxmax - dxmin);
    const int nrows = dzmax - dzmin + 1;
    std::vector<uint64_t> P0((size_t)nrows * width), P1((size_t)nrows * width);
    std::vector<size_t> rowOff(f.size());

    for (long ya = b.y0; ya <= b.y1; ya += slabH)
    {
        const long yb = std::min<long>(b.y1, ya + slabH - 1);
        const int nys = (int)(yb - ya + 1);
        const uint64_t valid = nys == 64 ? ~0ull : ((1ull << nys) - 1);
        const long wlo = ya + dymin, whi = yb + dymax;
        uint64_t need = 0;
        for (auto &k : f) need |= valid << (k.dy - dymin);
        auto fillRow = [&](long pz)
        {
            uint64_t *r0 = &P0[(size_t)pmod(pz, nrows) * width], *r1 = &P1[(size_t)pmod(pz, nrows) * width];
            // first lattice px >= pxmin on this row
            int res = pmod((long)L.a * pxmin + (long)L.b * pz, L.s);
            for (long px = pxmin; px < pxmin + width; px++)
            {
                if (res == 0)
                    nField += (uint64_t)fillColumn<V>((int)px, (int)pz, wlo, whi, need, r0[px - pxmin], r1[px - pxmin]);
                res += L.a; if (res >= L.s) res -= L.s;
            }
        };
        for (long z = b.z0; z <= b.z1; z++)
        {
            if (z == b.z0) for (long pz = z + dzmin; pz <= z + dzmax; pz++) fillRow(pz);
            else fillRow(z + dzmax);
            const int64_t hz = (int64_t)z * 116129781LL;
            for (size_t i = 0; i < f.size(); i++)
                rowOff[i] = (size_t)pmod(z + f[i].dz, nrows) * width + (size_t)(f[i].dx - pxmin);
            int r = pmod((long)L.a * b.x0 + (long)L.b * z, L.s) - L.a;
            for (long x = b.x0; x <= b.x1; x++)
            {
                r += L.a; if (r >= L.s) r -= L.s; if (r < 0) r += L.s;
                uint64_t M = valid;
                for (int i : on[r])
                {
                    const Blk &k = f[i];
                    const size_t at = rowOff[i] + (size_t)x;
                    const int sh = k.dy - dymin;
                    const uint64_t q0 = P0[at] >> sh, q1 = P1[at] >> sh;
                    uint64_t eq = (k.expected & 1) ? q0 : ~q0;
                    if (k.mask == 3) eq &= (k.expected & 2) ? q1 : ~q1;
                    M &= eq;
                    if (!M) break;
                }
                while (M)
                {
                    const int j = __builtin_ctzll(M); M &= M - 1;
                    const int y = (int)(ya + j);
                    const uint32_t hx = (uint32_t)x * 3129871u;
                    bool ok = true;
                    for (int i : off[r])
                    {
                        nFull++;
                        int v = outStage(V, coordMixFromParts(hx + f[i].xTerm, hz + f[i].zTerm, y + f[i].dy));
                        if ((v & f[i].mask) != f[i].expected) { ok = false; break; }
                    }
                    if (ok) out.emplace_back((int)x, y, (int)z);
                }
            }
        }
    }
    *fieldEvals = (double)nField; *fullEvals = (double)nFull;
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
static bool compare(const char *name, const Box &b, const std::vector<Blk> &f, Lattice L, bool timing)
{
    using clk = std::chrono::steady_clock;
    double e1, fe, fu;
    auto t0 = clk::now(); auto r = scanRef<V>(b, f, &e1);
    auto t1 = clk::now(); auto s = scanLattice<V>(b, f, L, &fe, &fu);
    auto t2 = clk::now();
    std::sort(r.begin(), r.end()); std::sort(s.begin(), s.end());
    bool ok = r == s;
    double cand = (double)(b.x1 - b.x0 + 1) * (b.y1 - b.y0 + 1) * (b.z1 - b.z0 + 1);
    double tr = std::chrono::duration<double>(t1 - t0).count(), ts = std::chrono::duration<double>(t2 - t1).count();
    if (timing || !ok)
        std::printf("%-26s v%d s=%-2d (a=%d,b=%d) %s matches=%zu | ref %.3fs %.2f full/cand | new %.3fs %.3f field + %.4f full /cand | %.1fx  (%.0f M cand/s)\n",
                    name, V, L.s, L.a, L.b, ok ? "IDENTICAL" : "MISMATCH", r.size(), tr, e1 / cand, ts, fe / cand, fu / cand, tr / ts, cand / ts / 1e6);
    return ok;
}

int main(int argc, char **argv)
{
    auto f = loadFormation(argc > 1 ? argv[1] : "formation_a.txt");
    bool ok = true;

    std::mt19937_64 rng(999);
    auto ri = [&](long lo, long hi) { return (int)(lo + (long)(rng() % (uint64_t)(hi - lo + 1))); };
    int trials = 0;
    for (int it = 0; it < 600; it++)
    {
        std::vector<Blk> g; int nb = ri(1, 7);
        for (int i = 0; i < nb; i++)
        {
            int side = ri(0, 2) == 0, x = ri(-5, 5), y = ri(-6, 6), z = ri(-5, 5), r = side ? ri(0, 1) : ri(0, 3);
            g.push_back({x, y, z, r, side ? 1 : 3, (uint32_t)x * 3129871u, (int64_t)z * 116129781LL});
        }
        int far = ri(0, 3) == 0;
        int cx = far ? ri(-29999900, 29999900) : ri(-200000, 200000), cz = far ? ri(-29999900, 29999900) : ri(-200000, 200000);
        int ya = ri(-2100, 2100), yb = ya + ri(0, 200);
        if (ri(0, 2) == 0) { ya = ri(-64, 319); yb = ri(ya, 319); }
        Box b{cx, cx + ri(0, 20), ya, yb, cz, cz + ri(0, 20)};
        int s = ri(1, 9), a, bb;
        do { a = ri(0, s - 1); bb = ri(0, s - 1); } while (s > 1 && std::gcd(std::gcd(a, bb), s) != 1);
        Lattice L{s, s == 1 ? 1 : a, s == 1 ? 0 : bb, 0};
        ok &= compare<0>("random", b, g, L, false); ok &= compare<1>("random", b, g, L, false);
        ok &= compare<2>("random", b, g, L, false); ok &= compare<3>("random", b, g, L, false);
        ok &= compare<4>("random", b, g, L, false);
        trials += 5;
    }
    std::printf("randomised equivalence: %d trials (random formations, boxes, lattices, all versions) -> %s\n", trials, ok ? "all identical" : "MISMATCH");

    Box w{-110200, -107200, -64, -40, -70500, -69000};     // W's y range
    Box full{-109200, -108200, -64, 319, -70000, -69500};  // full height
    Box flat{-116000, -101000, -54, -54, -74000, -66000};  // single level
    Lattice dense{1, 1, 0, 0};
    Lattice Ls = chooseLattice(f, 1.0, 16), Lf = chooseLattice(f, 3.0, 16);
    std::printf("chosen lattice (multi-level): s=%d a=%d b=%d modelled %.3f mul/cand; (single level): s=%d a=%d b=%d modelled %.3f\n",
                Ls.s, Ls.a, Ls.b, Ls.cost, Lf.s, Lf.a, Lf.b, Lf.cost);
    ok &= compare<0>("W slice bitplanes only", w, f, dense, true);
    ok &= compare<0>("W slice + lattice", w, f, Ls, true);
    ok &= compare<1>("W slice + lattice", w, f, Ls, true);
    ok &= compare<4>("W slice + lattice", w, f, Ls, true);
    ok &= compare<0>("full height bitplanes only", full, f, dense, true);
    ok &= compare<0>("full height + lattice", full, f, Ls, true);
    ok &= compare<0>("single level bitplanes", flat, f, dense, true);
    ok &= compare<0>("single level + lattice", flat, f, Lf, true);
    for (int s : {2, 3, 4, 5, 6, 8, 12})
    {
        Lattice best{s, 1, 0, 1e30};
        for (int a = 0; a < s; a++) for (int bb = 0; bb < s; bb++)
        {
            if (std::gcd(std::gcd(a, bb), s) != 1) continue;
            std::vector<double> surv(s, 1.0);
            for (auto &k : f) surv[pmod(-(long)a * k.dx - (long)bb * k.dz, s)] *= k.mask == 3 ? 0.25 : 0.5;
            double c = 0; for (double v : surv) c += v;
            if (c < best.cost) best = {s, a, bb, c};
        }
        ok &= compare<0>("W slice, forced s", w, f, best, true);
    }
    return ok ? 0 : 1;
}
