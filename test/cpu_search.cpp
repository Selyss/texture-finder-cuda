// Independent CPU implementation of the full search pipeline, for
// differential testing against the GPU on small volumes. Takes the same
// arguments as the main program and prints the same "Match found" lines
// (sorted), so outputs can be diffed directly. Deliberately simple and
// obviously correct: triple loop, no early-exit tricks beyond the block loop.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "blockinfo.cuh"
#include "parser.cuh"
#include "rotation.cuh"
#include "texture.cuh"

namespace
{
const char *DIRECTION_NAMES[4] = {"North", "West", "South", "East"};

int textureAt(bool modern, int x, int y, int z, int mod)
{
    return modern ? getTextureModern(x, y, z, mod) : getTextureLegacy(x, y, z, mod);
}
} // namespace

int main(int argc, char *argv[])
{
    if (argc != 10)
    {
        std::fprintf(stderr,
                     "Usage: %s <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction|all>\n",
                     argv[0]);
        return 1;
    }
    const int x_min = std::atoi(argv[1]), x_max = std::atoi(argv[2]);
    const int y_min = std::atoi(argv[3]), y_max = std::atoi(argv[4]);
    const int z_min = std::atoi(argv[5]), z_max = std::atoi(argv[6]);
    const bool modern = std::atoi(argv[7]) == MODERN_VERSION;

    std::vector<int> directions;
    const bool allMode = std::string(argv[9]) == "all";
    if (allMode)
        directions = {0, 1, 2, 3};
    else
        directions = {std::atoi(argv[9])};

    std::vector<BlockInfo> formation;
    try
    {
        formation = parseFormationFile(argv[8]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    struct Found
    {
        int x, y, z, dir;
    };
    std::vector<Found> found;

    for (int dir : directions)
    {
        std::vector<BlockInfo> blocks = formation;
        for (int r = 0; r < dir; r++)
            for (auto &b : blocks)
                rotateFormation(b);
        std::stable_partition(blocks.begin(), blocks.end(),
                              [](const BlockInfo &b) { return !b.isSide; });

        for (int x = x_min; x <= x_max; x++)
            for (int y = y_min; y <= y_max; y++)
                for (int z = z_min; z <= z_max; z++)
                {
                    bool ok = true;
                    for (const auto &b : blocks)
                        if (textureAt(modern, x + b.x, y + b.y, z + b.z,
                                      b.isSide ? MOD_SIDE : MOD_TOP_BOTTOM) != b.rotation)
                        {
                            ok = false;
                            break;
                        }
                    if (ok)
                        found.push_back({x, y, z, dir});
                }
    }

    std::sort(found.begin(), found.end(), [](const Found &a, const Found &b)
              { return a.x != b.x   ? a.x < b.x
                       : a.y != b.y ? a.y < b.y
                       : a.z != b.z ? a.z < b.z
                                    : a.dir < b.dir; });
    for (const auto &f : found)
    {
        std::printf("Match found at [%d, %d, %d]", f.x, f.y, f.z);
        if (allMode)
            std::printf(" facing %s", DIRECTION_NAMES[f.dir]);
        std::printf("\n");
    }
    std::printf("%zu match%s\n", found.size(), found.size() == 1 ? "" : "es");
    return 0;
}
