#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#define TF_ISATTY_STDERR() _isatty(_fileno(stderr))
#else
#include <unistd.h>
#define TF_ISATTY_STDERR() isatty(fileno(stderr))
#endif

#include "session.hpp"
#include "checkpoint.hpp"
#include "texture/rotation.hpp"

namespace
{

// Volumes above this are searched in z-band slices so the driver can report
// progress, stream matches, and checkpoint between slices.
constexpr unsigned long long SLICE_THRESHOLD = 1'000'000'000ull;
constexpr unsigned long long FIRST_SLICE_POSITIONS = 200'000'000ull;
constexpr double TARGET_SLICE_SECONDS = 2.0;

const char *DIRECTION_NAMES[4] = {"North", "West", "South", "East"};

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

SearchOutcome runSearchSession(const SearchRequest &request)
{
    const auto &bounds = request.bounds;
    const auto &formation = request.formation;
    const auto version = request.version;
    const auto &directions = request.directions;
    const auto &checkpointPath = request.checkpointPath;
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
        return {1};
    }

    const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
    const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
    const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
    if (nx > (~0ull) / ny || nx * ny > (~0ull) / nz)
    {
        std::cerr << "Error: search volume too large" << std::endl;
        return {1};
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
    ckpt.sliceFirst = allMode && sharedSearchEnabled(version);
    ckpt.nextZ = bounds.z_min; // progress baseline even when no checkpoint is used
    if (!checkpointPath.empty())
    {
        ckpt.signature = checkpointSignature(request);
        ckpt.path = checkpointPath;
        ckpt.active = true;
        try
        {
            if (ckpt.load())
            {
                if (ckpt.sliceFirst && (!allMode || ckpt.dirIndex != 0 ||
                    ckpt.nextZ < bounds.z_min || ckpt.nextZ > static_cast<long long>(bounds.z_max) + 1))
                    throw std::runtime_error("invalid slice-first checkpoint: " + ckpt.path);
                std::cout << "Resuming from checkpoint: direction index " << ckpt.dirIndex
                          << ", z = " << ckpt.nextZ << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return {1};
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
                           (unsigned long long)((long long)ckpt.nextZ - bounds.z_min) * rowsPerZ *
                               (ckpt.sliceFirst ? directions.size() : 1));

    // Adaptive slice sizing: aim for TARGET_SLICE_SECONDS per slice based on
    // the measured rate of the previous slice.
    double rateEMA = 0.0;

    struct Found
    {
        MatchResult m;
        int direction;
    };
    std::vector<Found> collected;

    const auto rotate = [&](int dir) {
        SearchFormation rotated{topsAndBottoms, sides};
        for (int r = 0; r < dir; ++r)
        {
            for (auto &b : rotated.tops)
                rotateFormation(b);
            for (auto &b : rotated.sides)
                rotateFormation(b);
        }
        return rotated;
    };
    const auto accept = [&](const DirectionResult &result, int dir) {
        acc.truncated |= result.truncated;
        acc.totalMatches += result.total;
        acc.listed += result.matches.size();
        if (!sliced)
        {
            for (const auto &match : result.matches)
                collected.push_back({match, dir});
        }
        else if (!result.matches.empty())
        {
            progress.clearLine();
            for (const auto &match : result.matches)
                printMatch(match, dir, allMode);
            std::cout.flush();
        }
    };
    const auto sliceEnd = [&](long long z, unsigned long long positionsPerRow) {
        const auto target = rateEMA > 0 ? static_cast<unsigned long long>(rateEMA * TARGET_SLICE_SECONDS)
                                        : FIRST_SLICE_POSITIONS;
        const auto rows = std::max(1ull, target / positionsPerRow);
        return rows <= static_cast<unsigned long long>(bounds.z_max - z)
            ? z + static_cast<long long>(rows) - 1 : bounds.z_max;
    };
    const auto recordSlice = [&](unsigned long long positions, double seconds,
                                 size_t dir, long long nextZ, bool workRemains) {
        acc.kernelSec += seconds;
        if (seconds > 0)
        {
            const double rate = static_cast<double>(positions) / seconds;
            rateEMA = rateEMA > 0 ? 0.5 * rateEMA + 0.5 * rate : rate;
        }
        progress.update(positions);
        ckpt.dirIndex = dir;
        ckpt.nextZ = nextZ;
        ckpt.acc = acc;
        ckpt.save();

        ++slicesDone;
        if (maxSlices < 0 || slicesDone < maxSlices || !workRemains)
            return false;
        progress.end();
        std::cout << "Stopping after " << slicesDone << " slice(s) (TF_MAX_SLICES)";
        if (ckpt.active)
            std::cout << "; state saved to " << ckpt.path;
        std::cout << std::endl;
        return true;
    };

    if (ckpt.sliceFirst)
    {
        std::vector<SearchFormation> formations;
        for (int dir : directions)
            formations.push_back(rotate(dir));
        const auto positionsPerRow = rowsPerZ * directions.size();
        for (long long z = ckpt.nextZ; z <= bounds.z_max;)
        {
            SearchBounds slice = bounds;
            const auto end = sliced ? sliceEnd(z, positionsPerRow) : bounds.z_max;
            slice.z_min = static_cast<int>(z);
            slice.z_max = static_cast<int>(end);
            const auto batch = runSharedSearch(slice, formations, version);
            for (size_t di = 0; di < batch.directions.size(); ++di)
                accept(batch.directions[di], directions[di]);
            if (sliced)
            {
                if (recordSlice((end - z + 1) * positionsPerRow, batch.kernelSeconds, 0,
                                end + 1, end < bounds.z_max))
                    return {3};
            }
            else
                acc.kernelSec += batch.kernelSeconds;
            z = end + 1;
        }
    }

    for (size_t di = ckpt.dirIndex; !ckpt.sliceFirst && di < directions.size(); ++di)
    {
        const int dir = directions[di];
        const auto rotated = rotate(dir);
        const auto &tops = rotated.tops;
        const auto &sids = rotated.sides;

        if (!sliced)
        {
            const auto result = runSearch(bounds, tops, sids, version);
            acc.kernelSec += result.kernelMs / 1000.0;
            accept(result, dir);
            continue;
        }

        long long z = (di == ckpt.dirIndex && ckpt.resumed) ? ckpt.nextZ : bounds.z_min;
        while (z <= bounds.z_max)
        {
            const auto end = sliceEnd(z, rowsPerZ);
            SearchBounds sub = bounds;
            sub.z_min = static_cast<int>(z);
            sub.z_max = static_cast<int>(end);
            const auto result = runSearch(sub, tops, sids, version);
            accept(result, dir);

            const auto positions = static_cast<unsigned long long>(end - z + 1) * rowsPerZ;
            z = end + 1;
            if (recordSlice(positions, result.kernelMs / 1000.0, di, z,
                            z <= bounds.z_max || di + 1 < directions.size()))
                return {3};
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

    return {0, acc.kernelSec};
}
