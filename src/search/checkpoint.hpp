#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct SearchRequest;

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
    bool sliceFirst = false;
    std::uint64_t signature = 0;
    std::size_t dirIndex = 0;
    long long nextZ = 0;
    Accum acc;

    void save() const;
    void finish() const;
    bool load();
};

std::uint64_t checkpointSignature(const SearchRequest &request);
