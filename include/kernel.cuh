#pragma once
#include <vector>
#include "blockinfo.cuh"

// Backend-neutral search interface. Exactly one backend implementation is
// linked in, selected by the build (see Makefile): src/kernel.cu (CUDA GPU),
// src/backend_metal.mm (Apple GPU), or src/backend_cpu.cpp (portable
// multithreaded C++). All backends compile the same include/texture.cuh and
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

// Searches the (inclusive) bounds for origins where every block of the
// formation matches the texture rotation the game would generate.
// On return, *kernelMs holds the compute time of the search itself,
// *totalMatches the true number of matching origins, and *truncated is set
// if that exceeded the result buffer (the returned vector then contains the
// first bufferful; tighten the search in that case).
// Terminates the process with an error message on any backend failure.
std::vector<MatchResult> runSearch(const SearchBounds &bounds,
                                   const std::vector<BlockInfo> &topsAndBottoms,
                                   const std::vector<BlockInfo> &sides,
                                   int version,
                                   float *kernelMs = nullptr,
                                   bool *truncated = nullptr,
                                   unsigned long long *totalMatches = nullptr);
