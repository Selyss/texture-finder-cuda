// Generates a synthetic formation file whose blocks are guaranteed to match
// at a chosen origin, for end-to-end testing of the searcher. The emitted
// file is expressed in the "recreation frame" for the given direction: after
// the searcher rotates it <direction> times, it equals the ground truth.
//
// Usage: gen_formation <version 0-4> <direction 0-3> <ox> <oy> <oz> <count> <sidePercent> <seed>
// Output on stdout; '#' header lines record the parameters.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "blockinfo.cuh"
#include "rotation.cuh"
#include "texture.cuh"

int main(int argc, char *argv[])
{
    if (argc != 9)
    {
        std::fprintf(stderr,
                     "Usage: %s <version 0-4> <direction 0-3> <ox> <oy> <oz> <count> <sidePercent> <seed>\n",
                     argv[0]);
        return 1;
    }
    const int version = std::atoi(argv[1]);
    const int direction = std::atoi(argv[2]);
    const int ox = std::atoi(argv[3]);
    const int oy = std::atoi(argv[4]);
    const int oz = std::atoi(argv[5]);
    const int count = std::atoi(argv[6]);
    const int sidePercent = std::atoi(argv[7]);
    const unsigned seed = (unsigned)std::strtoul(argv[8], nullptr, 10);

    if (version < 0 || version >= NUM_VERSIONS || direction < 0 || direction > 3 || count < 1 ||
        count > MAX_FORMATION_BLOCKS || sidePercent < 0 || sidePercent > 100)
    {
        std::fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dxz(-8, 8), dy(-2, 3), pct(0, 99);

    // Ground truth in the world frame: real texture values at origin+offset.
    std::set<std::tuple<int, int, int>> used;
    std::vector<BlockInfo> truth;
    while ((int)truth.size() < count)
    {
        int bx = dxz(rng), by = dy(rng), bz = dxz(rng);
        if (!used.insert({bx, by, bz}).second)
            continue;
        BlockInfo b;
        b.x = bx;
        b.y = by;
        b.z = bz;
        b.isSide = pct(rng) < sidePercent;
        const int mod = b.isSide ? MOD_SIDE : MOD_TOP_BOTTOM;
        b.rotation = getTextureForVersion(version, ox + bx, oy + by, oz + bz, mod);
        truth.push_back(b);
    }

    // The searcher applies rotateFormation <direction> times to the file it
    // reads; emitting truth rotated (4 - direction) times round-trips to the
    // world frame exactly.
    for (auto &b : truth)
        for (int r = 0; r < (4 - direction) % 4; r++)
            rotateFormation(b);

    std::printf("# synthetic formation: version=%d direction=%d origin=%d,%d,%d count=%d sidePercent=%d seed=%u\n",
                version, direction, ox, oy, oz, count, sidePercent, seed);
    for (const auto &b : truth)
        std::printf("%d %d %d %d %d\n", b.x, b.y, b.z, b.rotation, b.isSide ? 1 : 0);
    return 0;
}
