#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blockinfo.cuh"
#include "kernel.cuh"
#include "parser.cuh"
#include "rotation.cuh"

namespace
{

// Coordinates are validated against the world border plus slack; this also
// guarantees the kernel's grid-stride arithmetic cannot overflow int.
constexpr long long COORD_LIMIT = 30'001'000;
constexpr long long Y_LIMIT = 20'000;

const char *DIRECTION_NAMES[4] = {"North", "West", "South", "East"};

void usage(const char *prog)
{
    std::cerr << "Usage: " << prog
              << " <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction>\n"
              << "  version:   0 = 1.21.2+, 1 = 1.13 - 1.21.1\n"
              << "  direction: 0 = North, 1 = West, 2 = South, 3 = East, or 'all' to search\n"
              << "             every orientation and report which one matched\n";
}

int parseIntArg(const std::string &arg, const char *name, long long lo, long long hi)
{
    long long v;
    size_t pos = 0;
    try
    {
        v = std::stoll(arg, &pos);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(std::string(name) + " is not a number: " + arg);
    }
    if (pos != arg.size())
        throw std::runtime_error(std::string(name) + " is not a number: " + arg);
    if (v < lo || v > hi)
        throw std::runtime_error(std::string(name) + " must be in [" + std::to_string(lo) +
                                 ", " + std::to_string(hi) + "], got " + arg);
    return (int)v;
}

} // namespace

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    if (argc != 10)
    {
        usage(argv[0]);
        return 1;
    }

    SearchBounds bounds;
    int version;
    std::vector<int> directions;
    std::vector<BlockInfo> formation;
    try
    {
        bounds.x_min = parseIntArg(argv[1], "x_min", -COORD_LIMIT, COORD_LIMIT);
        bounds.x_max = parseIntArg(argv[2], "x_max", -COORD_LIMIT, COORD_LIMIT);
        bounds.y_min = parseIntArg(argv[3], "y_min", -Y_LIMIT, Y_LIMIT);
        bounds.y_max = parseIntArg(argv[4], "y_max", -Y_LIMIT, Y_LIMIT);
        bounds.z_min = parseIntArg(argv[5], "z_min", -COORD_LIMIT, COORD_LIMIT);
        bounds.z_max = parseIntArg(argv[6], "z_max", -COORD_LIMIT, COORD_LIMIT);
        if (bounds.x_min > bounds.x_max || bounds.y_min > bounds.y_max || bounds.z_min > bounds.z_max)
            throw std::runtime_error("min bound exceeds max bound");

        version = parseIntArg(argv[7], "version", 0, 1);

        if (std::string(argv[9]) == "all")
            directions = {0, 1, 2, 3};
        else
            directions = {parseIntArg(argv[9], "direction", 0, 3)};

        formation = parseFormationFile(argv[8]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        usage(argv[0]);
        return 1;
    }

    std::cout << "Version: " << (version == MODERN_VERSION ? "1.21.2+" : "1.13 - 1.21.1") << std::endl;
    if (directions.size() == 1)
        std::cout << "Facing: " << DIRECTION_NAMES[directions[0]] << std::endl;
    else
        std::cout << "Facing: trying all directions" << std::endl;

    std::vector<BlockInfo> topsAndBottoms, sides;
    for (const auto &info : formation)
        (info.isSide ? sides : topsAndBottoms).push_back(info);

    if (topsAndBottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS)
    {
        std::cerr << "Error: formation has too many blocks (max " << MAX_FORMATION_BLOCKS
                  << " per face type)" << std::endl;
        return 1;
    }

    // With too few blocks the search cannot be selective: warn up front how
    // many random positions are expected to match by pure chance.
    const double volume = (double)((long long)bounds.x_max - bounds.x_min + 1) *
                          (double)((long long)bounds.y_max - bounds.y_min + 1) *
                          (double)((long long)bounds.z_max - bounds.z_min + 1);
    const double expectedRandom = volume *
                                  std::pow(0.25, (double)topsAndBottoms.size()) *
                                  std::pow(0.5, (double)sides.size());
    if (expectedRandom > 100.0)
        std::cout << "Warning: formation is weak for this volume; ~" << expectedRandom
                  << " coincidental matches expected. Add more blocks or shrink the search area."
                  << std::endl;

    struct Found
    {
        MatchResult m;
        int direction;
    };
    std::vector<Found> found;
    float totalKernelMs = 0.f;
    bool anyTruncated = false;
    unsigned long long totalMatches = 0;

    for (int dir : directions)
    {
        std::vector<BlockInfo> tops = topsAndBottoms, sids = sides;
        for (int r = 0; r < dir; r++)
        {
            for (auto &b : tops)
                rotateFormation(b);
            for (auto &b : sids)
                rotateFormation(b);
        }

        float kernelMs = 0.f;
        bool truncated = false;
        unsigned long long dirTotal = 0;
        std::vector<MatchResult> matches =
            runSearch(bounds, tops, sids, version, &kernelMs, &truncated, &dirTotal);
        totalKernelMs += kernelMs;
        anyTruncated |= truncated;
        totalMatches += dirTotal;
        for (const auto &m : matches)
            found.push_back({m, dir});
    }

    std::sort(found.begin(), found.end(), [](const Found &a, const Found &b)
              { return a.m.x != b.m.x   ? a.m.x < b.m.x
                       : a.m.y != b.m.y ? a.m.y < b.m.y
                       : a.m.z != b.m.z ? a.m.z < b.m.z
                                        : a.direction < b.direction; });

    for (const auto &f : found)
    {
        std::cout << "Match found at [" << f.m.x << ", " << f.m.y << ", " << f.m.z << "]";
        if (directions.size() > 1)
            std::cout << " facing " << DIRECTION_NAMES[f.direction];
        std::cout << "\n";
    }

    if (found.empty())
        std::cout << "No matches found." << std::endl;
    else
        std::cout << totalMatches << (totalMatches == 1 ? " match" : " matches") << std::endl;
    if (anyTruncated)
        std::cout << "Warning: only the first " << found.size() << " of " << totalMatches
                  << " matches are listed; shrink the search area." << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel time: " << totalKernelMs / 1000.0 << " seconds" << std::endl;
    std::cout << "Search completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
