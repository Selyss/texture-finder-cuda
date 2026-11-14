#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>

#include "search/backend.hpp"
#include "texture/texture_rules.hpp"

namespace cuda_bitplanes
{
constexpr int STRIDE = 5;
constexpr int WORD_BITS = 64;
constexpr int THREADS = 256;
constexpr int DEFAULT_SLAB_ROWS = 4096;
using Word = unsigned long long;

struct Planes
{
    Word low, high;
};

struct Check
{
    int xOffset, ySlot, zOffset, rotation;
    bool isSide;
};

struct PlaneCheck
{
    size_t offset;
    int shift, rotation;
    bool isSide;
};

struct VerifyCheck
{
    int64_t zTerm;
    uint32_t xTerm;
    int yOffset, rotation, mask;
};

struct Layout
{
    SearchBounds bounds;
    int nx, ny, nz;
    int fieldX, fieldWidth, fieldWords, candidateWords;
    int ySlots, hashedRows;
    int64_t firstHashedZ;
};

enum class MatchMode { Full, Filter, NoMemory, Div32, NoTail, Search };

__host__ __device__ inline int residue(int64_t z)
{
    const int r = static_cast<int>(z % STRIDE);
    return r < 0 ? r + STRIDE : r;
}

inline void preparePlaneChecks(const Layout &slab, const std::vector<Check> &checks,
                               const int *cosetOffsets, std::vector<PlaneCheck> &result)
{
    for (int coset = 0; coset < STRIDE; ++coset)
    {
        const int64_t start = slab.bounds.z_min;
        const int64_t anchor = start + residue(coset - start);
        if (anchor > slab.bounds.z_max)
            continue;
        for (int i = cosetOffsets[coset]; i < cosetOffsets[coset + 1]; ++i)
        {
            const auto &check = checks[i];
            const size_t zIndex = (anchor + check.zOffset - slab.firstHashedZ) / STRIDE;
            const size_t level = check.ySlot;
            result[i] = {
                (level * slab.hashedRows + zIndex) * slab.fieldWords + check.xOffset / WORD_BITS,
                check.xOffset % WORD_BITS, check.rotation, check.isSide};
        }
    }
}

__host__ __device__ inline int ceilDiv(int value, int divisor)
{
    return value / divisor + (value % divisor != 0);
}

__host__ __device__ inline Word lowMask(int bits)
{
    return bits == WORD_BITS ? ~0ull : (1ull << bits) - 1;
}

// Offsets are sorted. Each candidate interval maps to a consecutive run of indices.
inline std::vector<int> compactLevels(const SearchBounds &bounds, const std::vector<int> &yOffsets)
{
    std::vector<int> planeY;
    int64_t nextY = std::numeric_limits<int>::min();
    for (int dy : yOffsets)
    {
        const int64_t first = static_cast<int64_t>(bounds.y_min) + dy;
        const int64_t last = static_cast<int64_t>(bounds.y_max) + dy;
        for (int64_t level = std::max(nextY, first); level <= last; ++level)
            planeY.push_back(static_cast<int>(level));
        nextY = last + 1;
    }
    return planeY;
}

template <bool WorldY, bool Pack32 = false>
__global__ void fillPlanes(Layout layout, const int *yOffsets, size_t count, Planes *planes)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const int word = index % layout.fieldWords;
    const size_t row = index / layout.fieldWords;
    const int zIndex = row % layout.hashedRows;
    const size_t level = row / layout.hashedRows;
    const int y = WorldY ? yOffsets[level]
                         : layout.bounds.y_min + static_cast<int>(level / layout.ySlots)
                           + yOffsets[level % layout.ySlots];
    const int64_t z = layout.firstHashedZ + static_cast<int64_t>(zIndex) * STRIDE;
    const int64_t xOffset = static_cast<int64_t>(word) * WORD_BITS;
    const int64_t remaining = layout.fieldWidth - xOffset;
    const int bits = remaining < WORD_BITS ? static_cast<int>(remaining) : WORD_BITS;
    const int64_t zTerm = z * 116129781LL;
    uint32_t xTerm = static_cast<uint32_t>(layout.fieldX + xOffset) * 3129871u;

    Planes pair{};
    if constexpr (Pack32)
    {
        if (bits > 0)
        {
            for (int half = 0; half < 2; ++half)
            {
                uint32_t low = 0, high = 0;
                for (int bit = 0; bit < 32; ++bit, xTerm += 3129871u)
                {
                    const uint32_t rotation = modernFromCoordRandom(
                        coordRandomFromParts(xTerm, zTerm, y), 4);
                    low = (low >> 1) | (rotation << 31);
                    high = (high >> 1) | ((rotation >> 1) << 31);
                }
                pair.low |= static_cast<Word>(low) << (half * 32);
                pair.high |= static_cast<Word>(high) << (half * 32);
            }
            pair.low &= lowMask(bits);
            pair.high &= lowMask(bits);
        }
    }
    else
    {
        for (int bit = 0; bit < bits; ++bit, xTerm += 3129871u)
        {
            const unsigned int rotation = modernFromCoordRandom(coordRandomFromParts(xTerm, zTerm, y), 4);
            pair.low |= static_cast<Word>(rotation & 1u) << bit;
            pair.high |= static_cast<Word>(rotation >> 1) << bit;
        }
    }
    planes[index] = pair;
}

__device__ inline unsigned int warpSum(unsigned int count)
{
    for (int step = 16; step > 0; step /= 2)
        count += __shfl_down_sync(0xffffffffu, count, step);
    return count;
}

__device__ inline Word syntheticWord(size_t index, unsigned int salt)
{
    // Mix adjacent addresses and give each check and plane its own stream.
    Word value = index ^ (static_cast<Word>(salt) << 32);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

template <MatchMode Mode, bool WorldY, bool Precomputed = false>
__global__ void matchWords(Layout layout,
                           const std::conditional_t<Precomputed, PlaneCheck, Check> *checks,
                           const int *cosetOffsets,
                           const Planes *planes, unsigned int *blockCounts, Word *masks,
                           int maskZRows,
                           const VerifyCheck *verify, const int *verifyOffsets,
                           MatchResult *matches, Word *totalMatches, Word capacity)
{
    const int segments = ceilDiv(layout.candidateWords, THREADS);
    using Row = std::conditional_t<Mode == MatchMode::Div32, unsigned int, size_t>;
    const Row row = blockIdx.x / segments;
    const int word = (blockIdx.x % segments) * THREADS + threadIdx.x;
    const int z = layout.bounds.z_min + static_cast<int>(row % layout.nz);
    const Row y = row / layout.nz;
    const int coset = residue(z);
    Word mask = 0;
    if (word < layout.candidateWords)
    {
        const int remaining = layout.nx - word * WORD_BITS;
        mask = lowMask(remaining < WORD_BITS ? remaining : WORD_BITS);
        size_t base = 0;
        if constexpr (Precomputed)
        {
            static_assert(WorldY, "Precomputed offsets use compact world-y levels");
            // The coset anchor is within the slab's first STRIDE rows.
            // This quotient is the number of strides since that anchor.
            const size_t zWithinSlab = row % layout.nz;
            base = (y * layout.hashedRows + zWithinSlab / STRIDE) * layout.fieldWords + word;
        }
        for (int i = cosetOffsets[coset]; i < cosetOffsets[coset + 1] && mask; ++i)
        {
            const auto check = checks[i];
            size_t index;
            int shift;
            if constexpr (Precomputed)
            {
                index = base + check.offset;
                shift = check.shift;
            }
            else
            {
                const size_t zIndex = (static_cast<int64_t>(z) + check.zOffset - layout.firstHashedZ) / STRIDE;
                const size_t level = WorldY ? static_cast<size_t>(check.ySlot) + y
                                            : static_cast<size_t>(y) * layout.ySlots + check.ySlot;
                const size_t fieldRow = level * layout.hashedRows + zIndex;
                index = fieldRow * layout.fieldWords + word + check.xOffset / WORD_BITS;
                shift = check.xOffset % WORD_BITS;
            }
            Word low = (Mode == MatchMode::NoMemory
                    ? syntheticWord(index, 2 * i) : planes[index].low) >> shift;
            if (shift)
                low |= (Mode == MatchMode::NoMemory
                        ? syntheticWord(index + 1, 2 * i) : planes[index + 1].low)
                        << (WORD_BITS - shift);
            mask &= (check.rotation & 1) ? low : ~low;
            if (!check.isSide)
            {
                Word high = (Mode == MatchMode::NoMemory
                    ? syntheticWord(index, 2 * i + 1) : planes[index].high) >> shift;
                if (shift)
                    high |= (Mode == MatchMode::NoMemory
                        ? syntheticWord(index + 1, 2 * i + 1) : planes[index + 1].high)
                        << (WORD_BITS - shift);
                mask &= (check.rotation & 2) ? high : ~high;
            }
        }
        if (masks)
            masks[(y * maskZRows + row % layout.nz) * layout.candidateWords + word] = mask;
    }

    if constexpr (Mode == MatchMode::Full || Mode == MatchMode::Search)
    {
        // Each surviving bit belongs to exactly one thread. Check the blocks outside its coset.
        const int candidateY = layout.bounds.y_min + static_cast<int>(y);
        const int64_t zTerm = static_cast<int64_t>(z) * 116129781LL;
        for (Word survivors = mask; survivors; survivors &= survivors - 1)
        {
            const int bit = __ffsll(survivors) - 1;
            const int x = layout.bounds.x_min + word * WORD_BITS + bit;
            const uint32_t xTerm = static_cast<uint32_t>(x) * 3129871u;
            bool passes = true;
            for (int i = verifyOffsets[coset]; i < verifyOffsets[coset + 1]; ++i)
            {
                const VerifyCheck check = verify[i];
                const int rotation = modernFromCoordRandom(
                    coordRandomFromParts(xTerm + check.xTerm, zTerm + check.zTerm,
                                         candidateY + check.yOffset), 4);
                if ((rotation & check.mask) != check.rotation)
                {
                    passes = false;
                    break;
                }
            }
            if (passes)
            {
                const Word index = atomicAdd(totalMatches, 1ull);
                if (index < capacity)
                    matches[index] = {x, candidateY, z};
            }
        }
    }

    if constexpr (Mode == MatchMode::NoTail)
    {
        // Keep every thread's filter work live without storing masks during timing.
        asm volatile("" :: "l"(mask));
    }
    else if constexpr (Mode != MatchMode::Search)
    {
        __shared__ unsigned int warpCounts[THREADS / 32];
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        const unsigned int count = warpSum(__popcll(mask));
        if (!lane)
            warpCounts[warp] = count;
        __syncthreads();
        if (!warp)
        {
            const unsigned int total = warpSum(lane < THREADS / 32 ? warpCounts[lane] : 0);
            if (!lane)
                blockCounts[blockIdx.x] = total;
        }
    }
}

} // namespace cuda_bitplanes
