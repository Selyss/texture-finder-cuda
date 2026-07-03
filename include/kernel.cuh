#pragma once
#include <vector>
#include "blockinfo.cuh"

struct SearchBounds
{
    int x_min, x_max;
    int y_min, y_max;
    int z_min, z_max;
};

struct MatchResult
{
    int x, y, z;
};

// Searches the (inclusive) bounds for origins where every block of the
// formation matches the texture rotation the game would generate.
// On return, *kernelMs holds the kernel execution time, *totalMatches the
// true number of matching origins, and *truncated is set if that exceeded
// the result buffer (the returned vector then contains the first bufferful;
// tighten the search in that case).
// Terminates the process with an error message on any CUDA failure.
std::vector<MatchResult> runSearch(const SearchBounds &bounds,
                                   const std::vector<BlockInfo> &topsAndBottoms,
                                   const std::vector<BlockInfo> &sides,
                                   int version,
                                   float *kernelMs = nullptr,
                                   bool *truncated = nullptr,
                                   unsigned long long *totalMatches = nullptr);
