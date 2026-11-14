#pragma once
#include <vector>
#include "texture/formation.hpp"

// Backend-neutral search interface. Exactly one backend implementation is
// linked in, selected by the build (see Makefile): src/backends/cuda/search.cu (CUDA GPU),
// src/backends/metal/search.mm (Apple GPU), or src/backends/cpu/search.cpp (portable
// multithreaded C++). All backends compile the same src/texture/texture_rules.hpp and
// must be behaviorally identical; test/e2e.sh validates whichever backend
// the current binary carries.

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

// If a search legitimately produces more matches than this, the formation is
// far too small to identify a location anyway; backends report the true
// total but store only this many results.
constexpr unsigned int MAX_MATCHES = 1u << 20;

// Human-readable name of the linked backend ("CUDA", "Metal", "CPU").
const char *searchBackendName();

struct SearchFormation
{
    std::vector<BlockInfo> tops, sides;
};

struct DirectionResult
{
    std::vector<MatchResult> matches;
    unsigned long long total = 0;
    bool truncated = false;
    float kernelMs = 0.0f; // Set for an independently timed search.
};

struct SharedSearchResult
{
    std::vector<DirectionResult> directions;
    double kernelSeconds = 0.0;
};

// Searches inclusive bounds. The total includes matches beyond the output cap.
DirectionResult runSearch(const SearchBounds &bounds,
                          const std::vector<BlockInfo> &topsAndBottoms,
                          const std::vector<BlockInfo> &sides, int version);

bool sharedSearchEnabled(int version);
SharedSearchResult runSharedSearch(const SearchBounds &bounds,
                                   const std::vector<SearchFormation> &formations, int version);
