#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define TF_ISATTY_STDERR() _isatty(_fileno(stderr))
#else
#include <unistd.h>
#define TF_ISATTY_STDERR() isatty(fileno(stderr))
#endif

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

// Volumes above this are searched in z-band slices so the driver can report
// progress, stream matches, and checkpoint between slices.
constexpr unsigned long long SLICE_THRESHOLD = 1'000'000'000ull;
constexpr unsigned long long FIRST_SLICE_POSITIONS = 200'000'000ull;
constexpr double TARGET_SLICE_SECONDS = 2.0;

const char *DIRECTION_NAMES[4] = {"North", "West", "South", "East"};

void usage(const char *prog)
{
    std::cerr << "Usage: " << prog
              << " <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction>"
              << " [--checkpoint <state-file>]\n"
              << "  version:   0 = 1.21.2+, 1 = 1.13 - 1.21.1, 2 = <=1.12.2,\n"
              << "             3 = Sodium 1.0-4.1 (MC 1.16-1.18.2), 4 = Sodium 4.2-4.8 (MC 1.19-1.19.3)\n"
              << "  direction: 0 = North, 1 = West, 2 = South, 3 = East, or 'all' to search\n"
              << "             every orientation and report which one matched\n"
              << "  --checkpoint: persist progress after every slice; rerunning the same\n"
              << "             command resumes an interrupted search from the saved state\n";
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

uint64_t fnv1a(const void *data, size_t n, uint64_t h = 0xCBF29CE484222325ull)
{
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++)
    {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

struct Accum
{
    unsigned long long totalMatches = 0;
    unsigned long long listed = 0;
    double kernelSec = 0.0;
    bool truncated = false;
};

// Checkpoint state: which direction index is in flight, the next unsearched
// z within it, and the accumulated counters. Guarded by a signature over the
// search arguments and the formation file content so a stale or mismatched
// state file can never silently corrupt a different search.
struct Checkpoint
{
    std::string path;
    bool active = false;
    bool resumed = false;
    uint64_t signature = 0;
    size_t dirIndex = 0;
    long long nextZ = 0;
    Accum acc;

    void save() const
    {
        if (!active)
            return;
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            f << "tfcp1 " << std::hex << signature << std::dec << " " << dirIndex << " "
              << nextZ << " " << acc.totalMatches << " " << acc.listed << " "
              << acc.kernelSec << " " << (acc.truncated ? 1 : 0) << "\n";
        }
        std::rename(tmp.c_str(), path.c_str());
    }

    void finish() const
    {
        if (active)
            std::remove(path.c_str());
    }

    // Returns false if no usable state exists; throws on signature mismatch.
    bool load()
    {
        std::ifstream f(path);
        if (!f)
            return false;
        std::string magic;
        uint64_t sig = 0;
        int trunc = 0;
        f >> magic >> std::hex >> sig >> std::dec >> dirIndex >> nextZ >>
            acc.totalMatches >> acc.listed >> acc.kernelSec >> trunc;
        if (!f || magic != "tfcp1")
            throw std::runtime_error("checkpoint file is corrupt: " + path);
        if (sig != signature)
            throw std::runtime_error(
                "checkpoint file belongs to a different search (bounds/version/"
                "direction/formation changed): " + path + " - delete it to start over");
        acc.truncated = trunc != 0;
        resumed = true;
        return true;
    }
};

struct Progress
{
    unsigned long long done = 0;
    unsigned long long total = 0;
    bool enabled = false;
    bool tty = false;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point lastPrint;
    bool linePending = false;

    void begin(unsigned long long totalPositions, unsigned long long alreadyDone)
    {
        total = totalPositions;
        done = alreadyDone;
        enabled = true;
        tty = TF_ISATTY_STDERR() != 0;
        start = lastPrint = std::chrono::steady_clock::now();
    }

    void update(unsigned long long positions)
    {
        if (!enabled)
            return;
        done += positions;
        const auto now = std::chrono::steady_clock::now();
        const double sinceLast = std::chrono::duration<double>(now - lastPrint).count();
        if (sinceLast < (tty ? 0.2 : 5.0) && done < total)
            return;
        lastPrint = now;

        const double elapsed = std::chrono::duration<double>(now - start).count();
        const double rate = elapsed > 0 ? (double)done / elapsed : 0;
        const double pct = total > 0 ? 100.0 * (double)done / (double)total : 100.0;
        char eta[64] = "--:--:--";
        if (rate > 0 && done < total)
        {
            long long s = (long long)((double)(total - done) / rate);
            std::snprintf(eta, sizeof(eta), "%02lld:%02lld:%02lld", s / 3600, s / 60 % 60, s % 60);
        }
        char line[160];
        std::snprintf(line, sizeof(line),
                      "Progress: %5.1f%% | %.3g/%.3g positions | %.3g/s | ETA %s",
                      pct, (double)done, (double)total, rate, eta);
        if (tty)
        {
            std::fprintf(stderr, "\r%-100s", line);
            linePending = true;
        }
        else
            std::fprintf(stderr, "%s\n", line);
        std::fflush(stderr);
    }

    // Keeps streamed stdout matches from colliding with the \r status line.
    void clearLine()
    {
        if (enabled && tty && linePending)
        {
            std::fprintf(stderr, "\r%-100s\r", "");
            std::fflush(stderr);
            linePending = false;
        }
    }

    void end()
    {
        if (enabled && tty && linePending)
            std::fprintf(stderr, "\n");
    }
};

void printMatch(const MatchResult &m, int direction, bool allMode)
{
    std::cout << "Match found at [" << m.x << ", " << m.y << ", " << m.z << "]";
    if (allMode)
        std::cout << " facing " << DIRECTION_NAMES[direction];
    std::cout << "\n";
}

} // namespace

int main(int argc, char *argv[])
{
    auto startWall = std::chrono::high_resolution_clock::now();

    std::string checkpointPath;
    if (argc == 12 && std::string(argv[10]) == "--checkpoint")
    {
        checkpointPath = argv[11];
        argc = 10;
    }
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

        version = parseIntArg(argv[7], "version", 0, NUM_VERSIONS - 1);

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

    std::cout << "Backend: " << searchBackendName() << std::endl;
    const char *VERSION_NAMES[NUM_VERSIONS] = {"1.21.2+", "1.13 - 1.21.1", "<=1.12.2",
                                               "Sodium 1.0-4.1 (MC 1.16 - 1.18.2)",
                                               "Sodium 4.2-4.8 (MC 1.19 - 1.19.3)"};
    std::cout << "Version: " << VERSION_NAMES[version] << std::endl;
    const bool allMode = directions.size() > 1;
    if (!allMode)
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

    const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
    const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
    const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
    if (nx > (~0ull) / ny || nx * ny > (~0ull) / nz)
    {
        std::cerr << "Error: search volume too large" << std::endl;
        return 1;
    }
    const unsigned long long rowsPerZ = nx * ny;
    const double volume = (double)nx * (double)ny * (double)nz;

    // Uniqueness lint: each top block divides coincidental matches by 4,
    // each side block by 2. Report exactly how much constraint is missing.
    const double haveBits = 2.0 * (double)topsAndBottoms.size() + (double)sides.size();
    const double needBits = std::log2(volume) + 6.64; // expected <= ~0.01 coincidences
    const double expectedRandom = volume * std::pow(2.0, -haveBits);
    if (needBits > haveBits)
    {
        const int moreTops = (int)std::ceil((needBits - haveBits) / 2.0);
        std::cout << "Warning: formation is weak for this volume; ~" << expectedRandom
                  << " coincidental matches expected. Add ~" << moreTops
                  << " more top-face blocks (or " << 2 * moreTops
                  << " side blocks) for a unique result." << std::endl;
    }

    Checkpoint ckpt;
    ckpt.nextZ = bounds.z_min; // progress baseline even when no checkpoint is used
    if (!checkpointPath.empty())
    {
        std::string dirsKey;
        for (int d : directions)
            dirsKey += std::to_string(d) + ",";
        std::ostringstream sig;
        sig << bounds.x_min << " " << bounds.x_max << " " << bounds.y_min << " " << bounds.y_max
            << " " << bounds.z_min << " " << bounds.z_max << " " << version << " " << dirsKey << "\n";
        for (const auto &b : formation)
            sig << b.x << " " << b.y << " " << b.z << " " << b.rotation << " " << b.isSide << "\n";
        const std::string s = sig.str();
        ckpt.signature = fnv1a(s.data(), s.size());
        ckpt.path = checkpointPath;
        ckpt.active = true;
        try
        {
            if (ckpt.load())
                std::cout << "Resuming from checkpoint: direction index " << ckpt.dirIndex
                          << ", z = " << ckpt.nextZ << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    // Debug/test hook: stop (exit 3) after N slices, leaving the checkpoint
    // behind, so resume can be exercised deterministically.
    long long maxSlices = -1;
    if (const char *ms = std::getenv("TF_MAX_SLICES"))
        maxSlices = std::atoll(ms);
    long long slicesDone = 0;

    const unsigned long long volumePerDir = rowsPerZ * nz;
    const bool sliced = ckpt.active || volumePerDir > SLICE_THRESHOLD;

    Accum acc = ckpt.acc;
    Progress progress;
    if (sliced)
        progress.begin(volumePerDir * directions.size(),
                       (unsigned long long)ckpt.dirIndex * volumePerDir +
                           (unsigned long long)((long long)ckpt.nextZ - bounds.z_min) * rowsPerZ);

    // Adaptive slice sizing: aim for TARGET_SLICE_SECONDS per slice based on
    // the measured rate of the previous slice.
    double rateEMA = 0.0;

    struct Found
    {
        MatchResult m;
        int direction;
    };
    std::vector<Found> collected;

    for (size_t di = ckpt.dirIndex; di < directions.size(); di++)
    {
        const int dir = directions[di];
        std::vector<BlockInfo> tops = topsAndBottoms, sids = sides;
        for (int r = 0; r < dir; r++)
        {
            for (auto &b : tops)
                rotateFormation(b);
            for (auto &b : sids)
                rotateFormation(b);
        }

        if (!sliced)
        {
            float kernelMs = 0.f;
            bool truncated = false;
            unsigned long long dirTotal = 0;
            std::vector<MatchResult> matches =
                runSearch(bounds, tops, sids, version, &kernelMs, &truncated, &dirTotal);
            acc.kernelSec += kernelMs / 1000.0;
            acc.truncated |= truncated;
            acc.totalMatches += dirTotal;
            acc.listed += matches.size();
            for (const auto &m : matches)
                collected.push_back({m, dir});
            continue;
        }

        long long z = (di == ckpt.dirIndex && ckpt.resumed) ? ckpt.nextZ : bounds.z_min;
        while (z <= bounds.z_max)
        {
            const unsigned long long targetPositions =
                rateEMA > 0 ? (unsigned long long)(rateEMA * TARGET_SLICE_SECONDS)
                            : FIRST_SLICE_POSITIONS;
            unsigned long long sliceZ = targetPositions / (rowsPerZ ? rowsPerZ : 1);
            if (sliceZ < 1)
                sliceZ = 1;
            const long long zEnd =
                (unsigned long long)(bounds.z_max - z) >= sliceZ ? z + (long long)sliceZ - 1
                                                                 : bounds.z_max;

            SearchBounds sub = bounds;
            sub.z_min = (int)z;
            sub.z_max = (int)zEnd;

            float kernelMs = 0.f;
            bool truncated = false;
            unsigned long long dirTotal = 0;
            std::vector<MatchResult> matches =
                runSearch(sub, tops, sids, version, &kernelMs, &truncated, &dirTotal);

            acc.kernelSec += kernelMs / 1000.0;
            acc.truncated |= truncated;
            acc.totalMatches += dirTotal;
            acc.listed += matches.size();

            if (!matches.empty())
            {
                progress.clearLine();
                for (const auto &m : matches)
                    printMatch(m, dir, allMode);
                std::cout.flush();
            }

            const unsigned long long slicePositions =
                (unsigned long long)(zEnd - z + 1) * rowsPerZ;
            const double sliceSec = kernelMs / 1000.0;
            if (sliceSec > 0)
            {
                const double rate = (double)slicePositions / sliceSec;
                rateEMA = rateEMA > 0 ? 0.5 * rateEMA + 0.5 * rate : rate;
            }
            progress.update(slicePositions);

            z = zEnd + 1;
            ckpt.dirIndex = di;
            ckpt.nextZ = z;
            ckpt.acc = acc;
            ckpt.save();

            slicesDone++;
            const bool workRemains = z <= bounds.z_max || di + 1 < directions.size();
            if (maxSlices >= 0 && slicesDone >= maxSlices && workRemains)
            {
                progress.end();
                std::cout << "Stopping after " << slicesDone << " slice(s) (TF_MAX_SLICES)";
                if (ckpt.active)
                    std::cout << "; state saved to " << ckpt.path;
                std::cout << std::endl;
                return 3;
            }
        }
        ckpt.dirIndex = di + 1;
        ckpt.nextZ = bounds.z_min;
        ckpt.acc = acc;
        ckpt.save();
    }

    progress.end();

    if (!sliced)
    {
        std::sort(collected.begin(), collected.end(), [](const Found &a, const Found &b)
                  { return a.m.x != b.m.x   ? a.m.x < b.m.x
                           : a.m.y != b.m.y ? a.m.y < b.m.y
                           : a.m.z != b.m.z ? a.m.z < b.m.z
                                            : a.direction < b.direction; });
        for (const auto &f : collected)
            printMatch(f.m, f.direction, allMode);
    }

    if (acc.totalMatches == 0)
        std::cout << "No matches found." << std::endl;
    else
        std::cout << acc.totalMatches << (acc.totalMatches == 1 ? " match" : " matches")
                  << std::endl;
    if (acc.truncated)
        std::cout << "Warning: only the first " << acc.listed << " of " << acc.totalMatches
                  << " matches are listed; shrink the search area." << std::endl;

    ckpt.finish();

    auto endWall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endWall - startWall;
    std::cout << "Kernel time: " << acc.kernelSec << " seconds" << std::endl;
    std::cout << "Search completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
