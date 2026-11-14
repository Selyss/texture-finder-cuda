#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/bitplanes.cuh"
#include "texture/formation_io.hpp"
#include "texture/rotation.hpp"
#include "texture/texture_rules.hpp"

namespace
{
using namespace cuda_bitplanes;
constexpr int SAMPLES = 9;
using Counts = std::array<Word, STRIDE>;

constexpr std::array matchModes{MatchMode::Full, MatchMode::Filter, MatchMode::NoMemory,
                               MatchMode::Div32, MatchMode::NoTail};
constexpr std::array modeNames{"full", "filter", "filter-nomem", "filter-div32", "filter-notail"};

struct Sample
{
    float baseline, fill, matchAndVerify;
    double baselineWall, prototypeWall;
};

struct SearchResult
{
    std::vector<MatchResult> matches;
    Word total = 0;
};

int checkedInt(int64_t value)
{
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        throw std::runtime_error("Experiment bounds exceed the supported integer range");
    return static_cast<int>(value);
}

size_t product(size_t a, size_t b)
{
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("Experiment size overflows size_t");
    return a * b;
}

unsigned int gridSize(size_t count)
{
    if (count > std::numeric_limits<int>::max())
        throw std::runtime_error("Experiment exceeds the CUDA grid size");
    return static_cast<unsigned int>(count);
}

void checkCuda(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
class DeviceArray
{
public:
    explicit DeviceArray(size_t count) : bytes_(product(count, sizeof(T)))
    {
        if (bytes_)
            checkCuda(cudaMalloc(&data_, bytes_));
    }

    ~DeviceArray() { if (data_) cudaFree(data_); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    T *get() const { return data_; }
    void upload(const T *source) { checkCuda(cudaMemcpy(data_, source, bytes_, cudaMemcpyHostToDevice)); }
    void download(T *destination) const
    {
        checkCuda(cudaMemcpy(destination, data_, bytes_, cudaMemcpyDeviceToHost));
    }
    void downloadPrefix(T *destination, size_t count) const
    {
        if (count)
            checkCuda(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T *data_ = nullptr;
    size_t bytes_;
};

class Event
{
public:
    Event() { checkCuda(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

template <class Launch>
float timeKernel(Launch launch)
{
    Event start, stop;
    checkCuda(cudaEventRecord(start.get()));
    launch();
    checkCuda(cudaGetLastError());
    checkCuda(cudaEventRecord(stop.get()));
    checkCuda(cudaEventSynchronize(stop.get()));
    float ms = 0;
    checkCuda(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    return ms;
}

struct Experiment
{
    Layout layout;
    int minZOffset, maxZOffset;
    std::vector<int> planeY;
    bool worldY;
    std::vector<Check> checks;
    std::array<int, STRIDE + 1> cosetOffsets{};
    std::vector<VerifyCheck> verify;
    std::array<int, STRIDE + 1> verifyOffsets{};
    const std::vector<BlockInfo> &formation;

    Experiment(const SearchBounds &bounds, const std::vector<BlockInfo> &blocks, bool compactY)
        : worldY(compactY), formation(blocks)
    {
        const int nx = checkedInt(static_cast<int64_t>(bounds.x_max) - bounds.x_min + 1);
        const int ny = checkedInt(static_cast<int64_t>(bounds.y_max) - bounds.y_min + 1);
        const int nz = checkedInt(static_cast<int64_t>(bounds.z_max) - bounds.z_min + 1);
        if (nx <= 0 || ny <= 0 || nz <= 0)
            throw std::runtime_error("Expected inclusive bounds in ascending order");

        auto [minX, maxX] = std::minmax_element(formation.begin(), formation.end(),
            [](const BlockInfo &a, const BlockInfo &b) { return a.x < b.x; });
        auto [minZ, maxZ] = std::minmax_element(formation.begin(), formation.end(),
            [](const BlockInfo &a, const BlockInfo &b) { return a.z < b.z; });
        minZOffset = minZ->z;
        maxZOffset = maxZ->z;
        const int fieldX = checkedInt(static_cast<int64_t>(bounds.x_min) + minX->x);
        const int endX = checkedInt(static_cast<int64_t>(bounds.x_max) + maxX->x);
        const int width = checkedInt(static_cast<int64_t>(endX) - fieldX + 1);
        checkedInt(static_cast<int64_t>(bounds.z_min) + minZOffset);
        checkedInt(static_cast<int64_t>(bounds.z_max) + maxZOffset);

        std::vector<int> yOffsets;
        for (const auto &block : formation)
        {
            checkedInt(static_cast<int64_t>(bounds.y_min) + block.y);
            checkedInt(static_cast<int64_t>(bounds.y_max) + block.y);
            yOffsets.push_back(block.y);
        }
        std::sort(yOffsets.begin(), yOffsets.end());
        yOffsets.erase(std::unique(yOffsets.begin(), yOffsets.end()), yOffsets.end());

        if (worldY)
        {
            planeY = compactLevels(bounds, yOffsets);
        }
        else
            planeY = yOffsets;


        for (int coset = 0; coset < STRIDE; ++coset)
        {
            for (const auto &block : formation)
            {
                if (residue(-static_cast<int64_t>(block.z)) != coset)
                {
                    verify.push_back({static_cast<int64_t>(block.z) * 116129781LL,
                                      static_cast<uint32_t>(block.x) * 3129871u,
                                      block.y, block.rotation, block.isSide ? 1 : 3});
                    continue;
                }
                const int key = worldY ? bounds.y_min + block.y : block.y;
                const int ySlot = std::lower_bound(planeY.begin(), planeY.end(), key) - planeY.begin();
                checks.push_back({block.x - minX->x, ySlot, block.z, block.rotation, block.isSide});
            }
            cosetOffsets[coset + 1] = static_cast<int>(checks.size());
            verifyOffsets[coset + 1] = static_cast<int>(verify.size());
        }
        layout = {bounds, nx, ny, nz, fieldX, width, ceilDiv(width, WORD_BITS) + 1,
                  ceilDiv(nx, WORD_BITS), checkedInt(planeY.size()), 0, 0};
    }

    size_t rows() const { return product(layout.ny, layout.nz); }
};

// Evaluate each origin directly, independently of GPU plane construction and extraction.
Word referenceMask(const Layout &layout, int y, int z, int word, const std::vector<BlockInfo> &blocks)
{
    const int start = word * WORD_BITS;
    const int bits = std::min(WORD_BITS, layout.nx - start);
    Word mask = 0;
    for (int bit = 0; bit < bits; ++bit)
    {
        const int x = layout.bounds.x_min + start + bit;
        bool passes = true;
        for (const auto &block : blocks)
        {
            const int mod = block.isSide ? MOD_SIDE : MOD_TOP_BOTTOM;
            if (getTextureModern(x + block.x, y + block.y, z + block.z, mod) != block.rotation)
            {
                passes = false;
                break;
            }
        }
        if (passes)
            mask |= 1ull << bit;
    }
    return mask;
}

Counts referenceCounts(const Experiment &experiment, unsigned int threads, const Word *masks = nullptr)
{
    const auto &layout = experiment.layout;
    const size_t rows = experiment.rows();
    auto checkRows = [&](size_t first, size_t last)
    {
        Counts counts{};
        std::vector<BlockInfo> selected;
        selected.reserve(experiment.formation.size());
        for (size_t row = first; row < last; ++row)
        {
            const int z = layout.bounds.z_min + static_cast<int>(row % layout.nz);
            const int y = layout.bounds.y_min + static_cast<int>(row / layout.nz);
            const int coset = residue(z);
            selected.clear();
            for (const auto &block : experiment.formation)
                if ((z + block.z) % STRIDE == 0)
                    selected.push_back(block);
            for (int word = 0; word < layout.candidateWords; ++word)
            {
                const Word expected = referenceMask(layout, y, z, word, selected);
                const size_t index = row * layout.candidateWords + word;
                if (masks && masks[index] != expected)
                    throw std::runtime_error("GPU/CPU mask mismatch at word " + std::to_string(index));
                counts[coset] += __builtin_popcountll(expected);
            }
        }
        return counts;
    };

    const size_t workers = std::min<size_t>(threads, rows);
    std::vector<std::future<Counts>> work;
    for (size_t i = 0; i < workers; ++i)
        work.push_back(std::async(std::launch::async, checkRows, rows * i / workers, rows * (i + 1) / workers));
    Counts totals{};
    for (auto &worker : work)
    {
        const auto counts = worker.get();
        for (int coset = 0; coset < STRIDE; ++coset)
            totals[coset] += counts[coset];
    }
    return totals;
}

class DeviceExperiment
{
public:
    DeviceExperiment(const Experiment &experiment, int slabRows, bool captureMasks,
                     size_t capacity = MAX_MATCHES, bool precompute = false, bool ownField = true,
                     const Experiment *fieldSource = nullptr)
        : layout_(fieldSource ? fieldSource->layout : experiment.layout),
          minZOffset_(fieldSource ? fieldSource->minZOffset : experiment.minZOffset),
          maxZOffset_(fieldSource ? fieldSource->maxZOffset : experiment.maxZOffset),
          slabRows_(std::min(slabRows, layout_.nz)),
          worldY_(experiment.worldY), precompute_(precompute),
          fieldWordsPerZ_(product(product(worldY_ ? 1 : layout_.ny, layout_.ySlots), layout_.fieldWords)),
          fieldCapacity_(product(fieldWordsPerZ_,
              (static_cast<int64_t>(slabRows_) + maxZOffset_ - minZOffset_ + STRIDE - 1) / STRIDE)),
          segments_(ceilDiv(layout_.candidateWords, THREADS)),
          matchCapacity_(gridSize(product(product(layout_.ny, slabRows_), segments_))),
          capacity_(capacity),
          planes_(ownField ? fieldCapacity_ : 0), checks_(precompute ? 0 : experiment.checks.size()),
          planeChecks_(precompute ? experiment.checks.size() : 0),
          slabChecks_(precompute ? experiment.checks.size() : 0), experiment_(experiment),
          offsets_(STRIDE + 1),
          yOffsets_(experiment.planeY.size()), counts_(matchCapacity_),
          masks_(captureMasks ? product(experiment.rows(), layout_.candidateWords) : 0),
          verify_(experiment.verify.size()), verifyOffsets_(STRIDE + 1),
          matches_(capacity_), totalMatches_(1)
    {
        if (!precompute_)
            checks_.upload(experiment.checks.data());
        offsets_.upload(experiment.cosetOffsets.data());
        yOffsets_.upload(experiment.planeY.data());
        verify_.upload(experiment.verify.data());
        verifyOffsets_.upload(experiment.verifyOffsets.data());
    }

    SearchResult run(Sample &sample, Counts *survivorCounts = nullptr,
                     MatchMode mode = MatchMode::Full, bool pack32 = false)
    {
        resetResults();
        sample.fill = 0;
        sample.matchAndVerify = 0;
        std::vector<unsigned int> hostCounts;
        if (survivorCounts && mode == MatchMode::NoTail)
            throw std::logic_error("No-tail mode does not produce block counts");
        if (survivorCounts)
        {
            survivorCounts->fill(0);
            hostCounts.resize(matchCapacity_);
        }

        for (int offset = 0; offset < layout_.nz;)
        {
            const Layout slab = slabAt(offset);
            sample.fill += fillSlab(slab, pack32);
            sample.matchAndVerify += matchSlab(slab, planes_.get(), offset, mode);
            if (survivorCounts)
                tallySlab(slab, hostCounts, *survivorCounts);
            offset += slab.nz;
        }
        if (mode != MatchMode::Full && mode != MatchMode::Search)
            return {};
        return results();
    }

    void resetResults() { checkCuda(cudaMemset(totalMatches_.get(), 0, sizeof(Word))); }
    const Planes *field() const { return planes_.get(); }

    Layout slabAt(int offset) const
    {
        Layout slab = layout_;
        slab.nz = std::min(slabRows_, layout_.nz - offset);
        slab.bounds.z_min = layout_.bounds.z_min + offset;
        slab.bounds.z_max = slab.bounds.z_min + (slab.nz - 1);
        // The halo supplies block lookups across slab boundaries. Candidates stay disjoint.
        const int64_t haloStart = static_cast<int64_t>(slab.bounds.z_min) + minZOffset_;
        const int64_t haloEnd = static_cast<int64_t>(slab.bounds.z_max) + maxZOffset_;
        slab.firstHashedZ = haloStart + residue(-haloStart);
        slab.hashedRows = slab.firstHashedZ <= haloEnd
            ? static_cast<int>((haloEnd - slab.firstHashedZ) / STRIDE + 1) : 0;
        return slab;
    }

    float fillSlab(const Layout &slab, bool pack32 = false)
    {
        const size_t fieldCount = fieldWordsPerZ_ * slab.hashedRows;
        const unsigned int fillGrid = gridSize(fieldCount / THREADS + (fieldCount % THREADS != 0));
        if (fillGrid)
            return timeKernel([&] {
                if (pack32)
                    fillPlanes<true, true><<<fillGrid, THREADS>>>(slab, yOffsets_.get(), fieldCount, planes_.get());
                else if (worldY_)
                    fillPlanes<true><<<fillGrid, THREADS>>>(slab, yOffsets_.get(), fieldCount, planes_.get());
                else
                    fillPlanes<false><<<fillGrid, THREADS>>>(slab, yOffsets_.get(), fieldCount, planes_.get());
            });
        return 0;
    }

    unsigned int blocksInSlab(const Layout &slab) const
    {
        return product(product(slab.ny, slab.nz), segments_);
    }

    float matchSlab(const Layout &slab, const Planes *field, int offset, MatchMode mode)
    {
        const unsigned int matchGrid = blocksInSlab(slab);
        Word *masks = masks_.get();
        if (masks)
            masks += static_cast<size_t>(offset) * slab.candidateWords;
        if (precompute_)
        {
            preparePlaneChecks(slab, experiment_.checks, experiment_.cosetOffsets.data(), slabChecks_);
            planeChecks_.upload(slabChecks_.data());
        }
        return timeKernel([&] {
            if (worldY_)
                launchMatch<true>(mode, slab, matchGrid, field, masks);
            else
                launchMatch<false>(mode, slab, matchGrid, field, masks);
        });
    }

    void tallySlab(const Layout &slab, std::vector<unsigned int> &hostCounts, Counts &counts)
    {
        const auto grid = blocksInSlab(slab);
        hostCounts.resize(grid);
        counts_.downloadPrefix(hostCounts.data(), grid);
        for (size_t block = 0; block < grid; ++block)
        {
            const int z = slab.bounds.z_min + static_cast<int>((block / segments_) % slab.nz);
            counts[residue(z)] += hostCounts[block];
        }
    }

    SearchResult results()
    {
        SearchResult result;
        totalMatches_.download(&result.total);
        result.matches.resize(std::min<Word>(capacity_, result.total));
        matches_.downloadPrefix(result.matches.data(), result.matches.size());
        std::sort(result.matches.begin(), result.matches.end(), [](const auto &a, const auto &b) {
            return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
        });
        return result;
    }

    void downloadMasks(Word *out) const { masks_.download(out); }
    size_t planeBytes() const { return planes_.get() ? fieldCapacity_ * sizeof(Planes) : 0; }
    int slabCount() const { return ceilDiv(layout_.nz, slabRows_); }

private:
    template <bool WorldY>
    void launchMatch(MatchMode mode, Layout slab, unsigned int grid, const Planes *field, Word *masks)
    {
        const auto launch = [&](auto tag) {
            if constexpr (WorldY)
            {
                if (precompute_)
                {
                    matchWords<decltype(tag)::value, true, true><<<grid, THREADS>>>(
                        slab, planeChecks_.get(), offsets_.get(), field, counts_.get(), masks,
                        layout_.nz, verify_.get(), verifyOffsets_.get(), matches_.get(),
                        totalMatches_.get(), capacity_);
                    return;
                }
            }
            matchWords<decltype(tag)::value, WorldY><<<grid, THREADS>>>(
                slab, checks_.get(), offsets_.get(), field, counts_.get(), masks, layout_.nz,
                verify_.get(), verifyOffsets_.get(), matches_.get(), totalMatches_.get(), capacity_);
        };
        switch (mode)
        {
        case MatchMode::Full: launch(std::integral_constant<MatchMode, MatchMode::Full>{}); break;
        case MatchMode::Filter: launch(std::integral_constant<MatchMode, MatchMode::Filter>{}); break;
        case MatchMode::NoMemory: launch(std::integral_constant<MatchMode, MatchMode::NoMemory>{}); break;
        case MatchMode::Div32: launch(std::integral_constant<MatchMode, MatchMode::Div32>{}); break;
        case MatchMode::NoTail: launch(std::integral_constant<MatchMode, MatchMode::NoTail>{}); break;
        case MatchMode::Search: launch(std::integral_constant<MatchMode, MatchMode::Search>{}); break;
        }
    }

    Layout layout_;
    int minZOffset_, maxZOffset_, slabRows_;
    bool worldY_, precompute_;
    size_t fieldWordsPerZ_, fieldCapacity_;
    int segments_;
    unsigned int matchCapacity_;
    size_t capacity_;
    DeviceArray<Planes> planes_;
    DeviceArray<Check> checks_;
    DeviceArray<PlaneCheck> planeChecks_;
    std::vector<PlaneCheck> slabChecks_;
    const Experiment &experiment_;
    DeviceArray<int> offsets_, yOffsets_;
    DeviceArray<unsigned int> counts_;
    DeviceArray<Word> masks_;
    DeviceArray<VerifyCheck> verify_;
    DeviceArray<int> verifyOffsets_;
    DeviceArray<MatchResult> matches_;
    DeviceArray<Word> totalMatches_;
};

bool sameMatch(const MatchResult &a, const MatchResult &b)
{
    return std::tie(a.x, a.y, a.z) == std::tie(b.x, b.y, b.z);
}

bool isMatch(const MatchResult &match, const SearchBounds &bounds,
             const std::vector<BlockInfo> &formation)
{
    if (match.x < bounds.x_min || match.x > bounds.x_max ||
        match.y < bounds.y_min || match.y > bounds.y_max ||
        match.z < bounds.z_min || match.z > bounds.z_max)
        return false;
    return std::all_of(formation.begin(), formation.end(), [&](const BlockInfo &block) {
        const int mod = block.isSide ? MOD_SIDE : MOD_TOP_BOTTOM;
        return getTextureModern(match.x + block.x, match.y + block.y, match.z + block.z, mod)
               == block.rotation;
    });
}

void compareResults(const SearchResult &actual, const SearchResult &expected,
                    const Experiment &experiment, size_t capacity = MAX_MATCHES)
{
    if (actual.total != expected.total ||
        actual.matches.size() != std::min<Word>(actual.total, capacity))
        throw std::runtime_error("Prototype/baseline result counts differ");
    if (actual.total <= capacity)
    {
        if (!std::equal(actual.matches.begin(), actual.matches.end(),
                        expected.matches.begin(), expected.matches.end(), sameMatch))
            throw std::runtime_error("Prototype/baseline match sets differ");
        return;
    }
    if (std::adjacent_find(actual.matches.begin(), actual.matches.end(), sameMatch) != actual.matches.end() ||
        !std::all_of(actual.matches.begin(), actual.matches.end(), [&](const MatchResult &match) {
            return isMatch(match, experiment.layout.bounds, experiment.formation);
        }))
        throw std::runtime_error("Prototype stored an invalid or duplicate match");
}

SearchResult baselineSearch(const SearchBounds &bounds, const std::vector<BlockInfo> &tops,
                            const std::vector<BlockInfo> &sides, float &kernelMs)
{
    auto result = runSearch(bounds, tops, sides, MODERN_VERSION);
    kernelMs = result.kernelMs;
    return {std::move(result.matches), result.total};
}

void verifySmallCase(bool worldY, int candidateLevels, bool sparseLevels, bool precompute = false)
{
    // Two x words plus a partial word, mixed faces, unaligned and multiword offsets,
    // negative coordinates, and a z interval covering every coset twice.
    const SearchBounds bounds{-70, 60, -2, -2 + candidateLevels - 1, -7, 4};
    std::vector<BlockInfo> formation{
        {0, -1, -1, 0, false}, {63, 2, 0, 1, false}, {65, 0, 1, 2, false},
        {-1, 0, 2, 1, true}, {3, 1, 3, 0, true}, {127, 0, 4, 3, false},
        {2, 1, 5, 0, true}, {7, 0, 6, 2, false}
    };
    if (sparseLevels)
    {
        const std::array offsets{-5, -2, 0, 50};
        for (size_t i = 0; i < formation.size(); ++i)
            formation[i].y = offsets[i % offsets.size()];
    }
    const MatchResult known{1, bounds.y_max, bounds.z_min + 5};
    std::vector<BlockInfo> tops, sides;
    for (auto &block : formation)
    {
        const int mod = block.isSide ? MOD_SIDE : MOD_TOP_BOTTOM;
        block.rotation = getTextureModern(known.x + block.x, known.y + block.y, known.z + block.z, mod);
        (block.isSide ? sides : tops).push_back(block);
    }
    const Experiment experiment(bounds, formation, worldY);
    const size_t expectedLevels = candidateLevels == 1 ? 4 : 11;
    if (worldY && sparseLevels && experiment.planeY.size() != expectedLevels)
        throw std::runtime_error("Unexpected compact level count for sparse offsets");
    DeviceExperiment device(experiment, sparseLevels ? 1 : 5, true, MAX_MATCHES, precompute);
    Sample sample{};
    Counts survivors;
    const auto actual = device.run(sample, &survivors);
    std::vector<Word> masks(experiment.rows() * experiment.layout.candidateWords);
    device.downloadMasks(masks.data());
    if (survivors != referenceCounts(experiment, 1, masks.data()))
        throw std::runtime_error("GPU/CPU count mismatch on the small case");
    const auto expected = baselineSearch(bounds, tops, sides, sample.baseline);
    compareResults(actual, expected, experiment);
    if (std::none_of(actual.matches.begin(), actual.matches.end(),
                     [&](const auto &match) { return sameMatch(match, known); }))
        throw std::runtime_error("Prototype missed the known synthetic match");
    for (const auto mode : {MatchMode::Filter, MatchMode::Div32, MatchMode::NoTail})
    {
        device.run(sample, nullptr, mode);
        device.downloadMasks(masks.data());
        referenceCounts(experiment, 1, masks.data());
    }
    std::puts("PASS small GPU/CPU masks and exact matches across slab boundaries");
    if (sparseLevels)
        return;


    const std::vector<BlockInfo> single{{0, 0, 0, 0, false}};
    const Experiment dense(bounds, single, worldY);
    if (worldY && dense.planeY.size() != static_cast<size_t>(candidateLevels))
        throw std::runtime_error("Unexpected compact level count for one offset");
    constexpr size_t smallCapacity = 3;
    DeviceExperiment limited(dense, 1, false, smallCapacity, precompute);
    const auto denseExpected = baselineSearch(bounds, single, {}, sample.baseline);
    if (denseExpected.total <= smallCapacity)
        throw std::runtime_error("Dense synthetic case did not exceed result capacity");
    compareResults(limited.run(sample), denseExpected, dense, smallCapacity);
    std::puts("PASS bounded result storage and true match count across slabs");
}

void runDiagnostics(const Experiment &experiment, int slabRows, const Counts &expected, bool allModes)
{
    const auto selected = [&](MatchMode mode) {
        return allModes || mode == MatchMode::Filter || mode == MatchMode::NoMemory;
    };
    DeviceExperiment device(experiment, slabRows, false);
    for (MatchMode mode : matchModes)
    {
        if (!selected(mode))
            continue;
        Sample warmup{};
        Counts counts{};
        device.run(warmup, mode == MatchMode::NoTail ? nullptr : &counts, mode);
        if (mode == MatchMode::NoMemory)
        {
            Word realTotal = 0, syntheticTotal = 0;
            for (int coset = 0; coset < STRIDE; ++coset)
            {
                realTotal += expected[coset];
                syntheticTotal += counts[coset];
            }
            std::printf("DIAG_SURVIVORS real=%llu synthetic=%llu\n", realTotal, syntheticTotal);
        }
        else if (mode != MatchMode::NoTail && counts != expected)
            throw std::runtime_error("Diagnostic survivor counts differ from CPU reference");
    }
    for (int sample = 0; sample < SAMPLES; ++sample)
    {
        for (size_t offset = 0; offset < matchModes.size(); ++offset)
        {
            const size_t index = (sample + offset) % matchModes.size();
            const MatchMode mode = matchModes[index];
            if (!selected(mode))
                continue;
            Sample timing{};
            device.run(timing, nullptr, mode);
            std::printf("DIAG %d %s fill_ms=%.6f match_ms=%.6f\n",
                        sample + 1, modeNames[index], timing.fill, timing.matchAndVerify);
        }
    }
}

void comparePlaneOffsets(const Experiment &experiment, int slabRows, const SearchResult &reference)
{
    if (!experiment.worldY)
        throw std::runtime_error("Offset comparison requires compact world-y planes");
    verifySmallCase(true, 4, false, true);
    verifySmallCase(true, 3, true, true);
    verifySmallCase(true, 1, true, true);

    Sample warmup{};
    Counts originalCounts{}, offsetCounts{};
    for (bool precompute : {false, true})
    {
        DeviceExperiment device(experiment, slabRows, false, MAX_MATCHES, precompute);
        auto &counts = precompute ? offsetCounts : originalCounts;
        compareResults(device.run(warmup, &counts), reference, experiment);
        device.run(warmup, nullptr, MatchMode::NoTail);
    }
    if (originalCounts != offsetCounts)
        throw std::runtime_error("Precomputed offsets changed survivor counts");
    for (int coset = 0; coset < STRIDE; ++coset)
        std::printf("COSET %d %llu\n", coset, offsetCounts[coset]);

    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    for (int i = 0; i < SAMPLES; ++i)
    {
        std::array<Sample, 2> search{}, filter{};
        for (int j = 0; j < 2; ++j)
        {
            const bool precompute = (i + j) % 2 != 0;
            auto &sample = search[precompute];
            SearchResult result;
            const auto start = Clock::now();
            {
                DeviceExperiment device(experiment, slabRows, false, MAX_MATCHES, precompute);
                result = device.run(sample, nullptr, MatchMode::Search);
            }
            sample.prototypeWall = Milliseconds(Clock::now() - start).count();
            compareResults(result, reference, experiment);

            DeviceExperiment device(experiment, slabRows, false, MAX_MATCHES, precompute);
            device.run(filter[precompute], nullptr, MatchMode::NoTail);
        }
        std::printf("OFFSET_PAIR %d %.6f %.6f %.6f %.6f %.6f %.6f\n", i + 1,
                    filter[0].matchAndVerify, filter[1].matchAndVerify,
                    search[0].matchAndVerify, search[1].matchAndVerify,
                    search[0].prototypeWall, search[1].prototypeWall);
    }
    std::puts("PASS precomputed offsets preserve masks, survivor counts and exact matches");
}

struct DirectionTiming
{
    float fill = 0, match = 0;
    double wall = 0;
};

void compareSharedField(const SearchBounds &bounds, const std::vector<BlockInfo> &formation,
                        int slabRows, bool checkMasks)
{
    constexpr int directions = 4;
    std::array<std::vector<BlockInfo>, directions> rotated;
    auto blocks = formation;
    std::vector<BlockInfo> unionBlocks;
    for (auto &direction : rotated)
    {
        direction = blocks;
        unionBlocks.insert(unionBlocks.end(), blocks.begin(), blocks.end());
        for (auto &block : blocks)
            rotateFormation(block);
    }
    const Experiment unionField(bounds, unionBlocks, true);
    std::array<std::optional<Experiment>, directions> independent, shared;
    std::array<SearchResult, directions> expected;
    std::array<Counts, directions> counts;
    std::array<std::vector<Word>, directions> masks;
    std::array<size_t, directions> individualBytes{};
    for (int dir = 0; dir < directions; ++dir)
    {
        const auto &direction = rotated[dir];
        independent[dir].emplace(bounds, direction, true);
        shared[dir].emplace(bounds, direction, true);
        for (auto &check : shared[dir]->checks)
            check.xOffset = checkedInt(static_cast<int64_t>(check.xOffset) +
                shared[dir]->layout.fieldX - unionField.layout.fieldX);
        const auto side = std::find_if(direction.begin(), direction.end(),
                                       [](const BlockInfo &block) { return block.isSide; });
        float baselineMs = 0;
        expected[dir] = baselineSearch(bounds, {direction.begin(), side},
                                       {side, direction.end()}, baselineMs);
        Sample sample{};
        DeviceExperiment device(*independent[dir], slabRows, checkMasks, MAX_MATCHES, true);
        compareResults(device.run(sample, &counts[dir]), expected[dir], *independent[dir]);
        individualBytes[dir] = device.planeBytes();
        if (checkMasks)
        {
            masks[dir].resize(independent[dir]->rows() * independent[dir]->layout.candidateWords);
            device.downloadMasks(masks[dir].data());
            referenceCounts(*independent[dir], 1, masks[dir].data());
        }
    }

    const auto run = [&](bool reuse, bool validate) {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        DirectionTiming time;
        if (reuse)
        {
            std::array<std::unique_ptr<DeviceExperiment>, directions> devices;
            for (int dir = 0; dir < directions; ++dir)
            {
                devices[dir] = std::make_unique<DeviceExperiment>(
                    *shared[dir], slabRows, validate && checkMasks, MAX_MATCHES, true, dir == 0, &unionField);
                devices[dir]->resetResults();
            }
            auto &owner = *devices.front();
            const auto mode = validate ? MatchMode::Full : MatchMode::Search;
            std::array<Counts, directions> actualCounts{};
            std::vector<unsigned int> scratch;
            for (int offset = 0; offset < unionField.layout.nz;)
            {
                const auto slab = owner.slabAt(offset);
                time.fill += owner.fillSlab(slab);
                for (int dir = 0; dir < directions; ++dir)
                {
                    time.match += devices[dir]->matchSlab(slab, owner.field(), offset, mode);
                    if (validate)
                        devices[dir]->tallySlab(slab, scratch, actualCounts[dir]);
                }
                offset += slab.nz;
            }
            for (int dir = 0; dir < directions; ++dir)
            {
                compareResults(devices[dir]->results(), expected[dir], *independent[dir]);
                if (validate && actualCounts[dir] != counts[dir])
                    throw std::runtime_error("Shared field changed survivor counts");
                if (validate && checkMasks)
                {
                    std::vector<Word> actual(masks[dir].size());
                    devices[dir]->downloadMasks(actual.data());
                    if (actual != masks[dir])
                        throw std::runtime_error("Shared field changed a survivor mask");
                }
            }
            if (validate && !checkMasks)
            {
                std::printf("SHARED_BYTES %zu %zu %zu %zu %zu\n", owner.planeBytes(),
                            individualBytes[0], individualBytes[1], individualBytes[2], individualBytes[3]);
                for (int dir = 0; dir < directions; ++dir)
                {
                    std::printf("DIRECTION %d matches=%llu survivors", dir, expected[dir].total);
                    for (Word count : actualCounts[dir])
                        std::printf(" %llu", count);
                    std::putchar('\n');
                }
            }
        }
        else
        {
            for (int dir = 0; dir < directions; ++dir)
            {
                Sample sample{};
                DeviceExperiment device(*independent[dir], slabRows, false, MAX_MATCHES, true);
                compareResults(device.run(sample, nullptr, MatchMode::Search), expected[dir], *independent[dir]);
                time.fill += sample.fill;
                time.match += sample.matchAndVerify;
            }
        }
        time.wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return time;
    };
    run(true, true);
    if (checkMasks)
        return;
    run(false, false);
    run(true, false);
    for (int i = 0; i < SAMPLES; ++i)
    {
        std::array<DirectionTiming, 2> pair;
        for (int j = 0; j < 2; ++j)
        {
            const bool reuse = (i + j) % 2 != 0;
            pair[reuse] = run(reuse, false);
        }
        std::printf("SHARED_PAIR %d %.6f %.6f %.6f %.6f %.6f %.6f\n", i + 1,
                    pair[0].fill, pair[1].fill, pair[0].match, pair[1].match,
                    pair[0].wall, pair[1].wall);
    }
    std::puts("PASS shared field preserves counts and exact results in all directions");
}

struct FusedCheck
{
    int shift, level, rotation;
};

struct FusedLayout
{
    SearchBounds bounds;
    int minX, minY, ySpan, xSpan, width, words, hashedRows;
    int64_t firstZ;
    int selectedZ[STRIDE], filterOffsets[STRIDE + 1], verifyOffsets[STRIDE + 1];
};

template <MatchMode Mode>
__device__ void fusedCandidates(Word mask, FusedLayout layout, int coset, int x, int y, int z,
                                int word, const VerifyCheck *verify, Word *masks,
                                Word &survivors, MatchResult *output, Word *total)
{
    if constexpr (Mode == MatchMode::NoTail)
        asm volatile("" :: "l"(mask));
    else
    {
        if constexpr (Mode == MatchMode::Full)
        {
            survivors += __popcll(mask);
            if (masks)
            {
                const size_t row = static_cast<size_t>(y - layout.bounds.y_min) *
                    (layout.bounds.z_max - layout.bounds.z_min + 1) + (z - layout.bounds.z_min);
                masks[row * layout.words + word] = mask;
            }
        }
        const int64_t zTerm = static_cast<int64_t>(z) * 116129781LL;
        for (; mask; mask &= mask - 1)
        {
            const int candidateX = x + __ffsll(mask) - 1;
            const uint32_t xTerm = static_cast<uint32_t>(candidateX) * 3129871u;
            bool passes = true;
            for (int i = layout.verifyOffsets[coset]; i < layout.verifyOffsets[coset + 1]; ++i)
            {
                const auto check = verify[i];
                if (modernFromCoordRandom(coordRandomFromParts(xTerm + check.xTerm,
                        zTerm + check.zTerm, y + check.yOffset), 4) != check.rotation)
                {
                    passes = false;
                    break;
                }
            }
            if (passes)
            {
                const Word index = atomicAdd(total, 1ull);
                if (index < MAX_MATCHES)
                    output[index] = {candidateX, y, z};
            }
        }
    }
}

template <MatchMode Mode>
__global__ void fusedSearch(FusedLayout layout, const FusedCheck *checks, const VerifyCheck *verify,
                            Word *masks, Word *counts, MatchResult *output, Word *total)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = index / layout.words;
    const int word = index % layout.words;
    const int x = layout.bounds.x_min + word * layout.width;
    const int bits = min(layout.width, layout.bounds.x_max - x + 1);
    const Word valid = lowMask(bits);
    Word survivors[STRIDE]{};
    if (row < layout.hashedRows)
    {
        const int64_t hashedZ = layout.firstZ + static_cast<int64_t>(row) * STRIDE;
        const int64_t zTerm = hashedZ * 116129781LL;
        const uint32_t firstX = static_cast<uint32_t>(x + layout.minX) * 3129871u;
        Planes previous{}, older{};
        for (int y = layout.bounds.y_min - layout.ySpan; y <= layout.bounds.y_max; ++y)
        {
            Planes current{};
            uint32_t xTerm = firstX;
            for (int bit = 0; bit < bits + layout.xSpan; ++bit, xTerm += 3129871u)
            {
                const unsigned int rotation = modernFromCoordRandom(
                    coordRandomFromParts(xTerm, zTerm, y + layout.minY + layout.ySpan), 4);
                current.low |= static_cast<Word>(rotation & 1u) << bit;
                current.high |= static_cast<Word>(rotation >> 1) << bit;
            }
            if (y >= layout.bounds.y_min)
            {
                for (int coset = 0; coset < STRIDE; ++coset)
                {
                    const int first = layout.filterOffsets[coset];
                    const int end = layout.filterOffsets[coset + 1];
                    const int64_t z = hashedZ - layout.selectedZ[coset];
                    if (first == end || z < layout.bounds.z_min || z > layout.bounds.z_max)
                        continue;
                    Word mask = valid;
                    for (int i = first; i < end && mask; ++i)
                    {
                        const auto check = checks[i];
                        const auto plane = check.level == 0 ? current : (check.level == 1 ? previous : older);
                        const Word low = plane.low >> check.shift;
                        const Word high = plane.high >> check.shift;
                        mask &= (check.rotation & 1) ? low : ~low;
                        mask &= (check.rotation & 2) ? high : ~high;
                    }
                    fusedCandidates<Mode>(mask, layout, coset, x, y, static_cast<int>(z), word,
                                          verify, masks, survivors[coset], output, total);
                }
            }
            older = previous;
            previous = current;
        }
    }
    else
    {
        const int z = layout.bounds.z_min + (row - layout.hashedRows);
        if (z > layout.bounds.z_max)
            return;
        const int coset = residue(z);
        if (layout.filterOffsets[coset] != layout.filterOffsets[coset + 1])
            return;
        for (int y = layout.bounds.y_min; y <= layout.bounds.y_max; ++y)
            fusedCandidates<Mode>(valid, layout, coset, x, y, z, word, verify, masks,
                                  survivors[coset], output, total);
    }
    if constexpr (Mode == MatchMode::Full)
        for (int coset = 0; coset < STRIDE; ++coset)
            if (survivors[coset])
                atomicAdd(counts + coset, survivors[coset]);
}

struct FusedExperiment
{
    FusedLayout layout{};
    std::vector<FusedCheck> filters;
    std::vector<VerifyCheck> verify;
    std::array<std::vector<BlockInfo>, STRIDE> selected;
    int gridRows;

    FusedExperiment(const SearchBounds &bounds, const std::vector<BlockInfo> &blocks)
    {
        layout.bounds = bounds;
        const auto [minX, maxX] = std::minmax_element(blocks.begin(), blocks.end(),
            [](const auto &a, const auto &b) { return a.x < b.x; });
        const auto [minY, maxY] = std::minmax_element(blocks.begin(), blocks.end(),
            [](const auto &a, const auto &b) { return a.y < b.y; });
        layout.minX = minX->x;
        layout.minY = minY->y;
        layout.xSpan = maxX->x - minX->x;
        layout.ySpan = maxY->y - minY->y;
        if (STRIDE != 5 || layout.xSpan >= WORD_BITS || layout.ySpan > 2 ||
            std::any_of(blocks.begin(), blocks.end(), [](const auto &b) { return b.isSide; }))
            throw std::runtime_error("Fused experiment requires stride 5, top faces, x span < 64 and y span <= 2");
        layout.width = WORD_BITS - layout.xSpan;
        layout.words = ceilDiv(checkedInt(static_cast<int64_t>(bounds.x_max) - bounds.x_min + 1), layout.width);
        std::map<int, int> byZ;
        for (const auto &b : blocks)
            ++byZ[b.z];
        std::array<int, STRIDE> best{};
        for (const auto &[z, count] : byZ)
        {
            const int coset = residue(-static_cast<int64_t>(z));
            if (count > best[coset])
            {
                best[coset] = count;
                layout.selectedZ[coset] = z;
            }
        }
        for (int coset = 0; coset < STRIDE; ++coset)
        {
            for (const auto &b : blocks)
            {
                if (best[coset] && b.z == layout.selectedZ[coset])
                {
                    filters.push_back({b.x - layout.minX, maxY->y - b.y, b.rotation});
                    selected[coset].push_back(b);
                }
                else
                    verify.push_back({static_cast<int64_t>(b.z) * 116129781LL,
                                      static_cast<uint32_t>(b.x) * 3129871u, b.y, b.rotation, 3});
            }
            layout.filterOffsets[coset + 1] = static_cast<int>(filters.size());
            layout.verifyOffsets[coset + 1] = static_cast<int>(verify.size());
        }
        const int64_t start = static_cast<int64_t>(bounds.z_min) + byZ.begin()->first;
        const int64_t end = static_cast<int64_t>(bounds.z_max) + byZ.rbegin()->first;
        layout.firstZ = start + residue(-start);
        layout.hashedRows = layout.firstZ <= end ? checkedInt((end - layout.firstZ) / STRIDE + 1) : 0;
        gridRows = layout.hashedRows;
        if (std::find(best.begin(), best.end(), 0) != best.end())
            gridRows = checkedInt(static_cast<int64_t>(gridRows) + bounds.z_max - bounds.z_min + 1);
    }

    size_t maskWords() const
    {
        const auto &b = layout.bounds;
        return product(product(layout.words, b.z_max - b.z_min + 1), b.y_max - b.y_min + 1);
    }
};

class DeviceFused
{
public:
    DeviceFused(const FusedExperiment &experiment, bool capture)
        : experiment_(experiment), filters_(experiment.filters.size()), verify_(experiment.verify.size()),
          masks_(capture ? experiment.maskWords() : 0), counts_(STRIDE), output_(MAX_MATCHES), total_(1)
    {
        filters_.upload(experiment.filters.data());
        if (!experiment.verify.empty())
            verify_.upload(experiment.verify.data());
    }

    template <MatchMode Mode>
    float run()
    {
        checkCuda(cudaMemset(total_.get(), 0, sizeof(Word)));
        checkCuda(cudaMemset(counts_.get(), 0, STRIDE * sizeof(Word)));
        const auto &layout = experiment_.layout;
        const size_t work = product(layout.words, experiment_.gridRows);
        if (!work)
            return 0;
        return timeKernel([&] {
            fusedSearch<Mode><<<gridSize((work + THREADS - 1) / THREADS), THREADS>>>(
                layout, filters_.get(), verify_.get(), masks_.get(), counts_.get(), output_.get(), total_.get());
        });
    }

    SearchResult results() const
    {
        SearchResult result;
        total_.download(&result.total);
        result.matches.resize(std::min<Word>(result.total, MAX_MATCHES));
        output_.downloadPrefix(result.matches.data(), result.matches.size());
        std::sort(result.matches.begin(), result.matches.end(), [](const auto &a, const auto &b) {
            return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
        });
        return result;
    }

    Counts counts() const
    {
        Counts result;
        counts_.download(result.data());
        return result;
    }

    void compareMasks() const
    {
        std::vector<Word> masks(experiment_.maskWords());
        masks_.download(masks.data());
        const auto &layout = experiment_.layout;
        const auto &b = layout.bounds;
        size_t index = 0;
        for (int y = b.y_min; y <= b.y_max; ++y)
            for (int z = b.z_min; z <= b.z_max; ++z)
                for (int word = 0; word < layout.words; ++word)
                {
                    Word expected = 0;
                    const int x = b.x_min + word * layout.width;
                    const int bits = std::min(layout.width, b.x_max - x + 1);
                    for (int bit = 0; bit < bits; ++bit)
                    {
                        const auto &blocks = experiment_.selected[residue(z)];
                        const bool passes = std::all_of(blocks.begin(), blocks.end(), [&](const auto &block) {
                            return getTextureModern(x + bit + block.x, y + block.y, z + block.z, 4) ==
                                   block.rotation;
                        });
                        if (passes)
                            expected |= 1ull << bit;
                    }
                    if (masks[index++] != expected)
                        throw std::runtime_error("Fused GPU mask differs from CPU oracle");
                }
    }

private:
    const FusedExperiment &experiment_;
    DeviceArray<FusedCheck> filters_;
    DeviceArray<VerifyCheck> verify_;
    DeviceArray<Word> masks_, counts_;
    DeviceArray<MatchResult> output_;
    DeviceArray<Word> total_;
};

void verifyFused(const SearchBounds &bounds, const std::vector<BlockInfo> &blocks,
                 const Counts *expectedCounts, bool capture)
{
    const Experiment planes(bounds, blocks, true);
    const FusedExperiment experiment(bounds, blocks);
    DeviceFused fused(experiment, capture);
    fused.run<MatchMode::Full>();
    if (capture)
        fused.compareMasks();
    if (expectedCounts && fused.counts() != *expectedCounts)
        throw std::runtime_error("Unexpected fused survivor counts");
    float ms = 0;
    compareResults(fused.results(), baselineSearch(bounds, blocks, {}, ms), planes);
}

Word fusedCpuCosetZero(const FusedExperiment &experiment, unsigned int threads)
{
    // Count independently of GPU packing. Only coset zero changes for the reference.
    const auto &layout = experiment.layout;
    const auto &bounds = layout.bounds;
    const int64_t first = static_cast<int64_t>(bounds.z_min) + residue(-static_cast<int64_t>(bounds.z_min));
    const size_t rows = first <= bounds.z_max ? (bounds.z_max - first) / STRIDE + 1 : 0;
    const auto &checks = experiment.selected[0];
    std::vector<std::future<Word>> workers;
    for (unsigned int i = 0; i < threads; ++i)
        workers.push_back(std::async(std::launch::async, [&, i] {
            Word count = 0;
            for (size_t row = rows * i / threads; row < rows * (i + 1) / threads; ++row)
            {
                const int z = static_cast<int>(first + static_cast<int64_t>(row) * STRIDE);
                for (int y = bounds.y_min; y <= bounds.y_max; ++y)
                    for (int x = bounds.x_min; x <= bounds.x_max; ++x)
                        if (std::all_of(checks.begin(), checks.end(), [&](const auto &b) {
                            return getTextureModern(x + b.x, y + b.y, z + b.z, 4) == b.rotation;
                        }))
                            ++count;
            }
            return count;
        }));
    Word total = 0;
    for (auto &worker : workers)
        total += worker.get();
    return total;
}

void timeFused(const char *label, const SearchBounds &bounds, const std::vector<BlockInfo> &blocks,
               int slabRows, unsigned int threads, bool reference)
{
    const FusedExperiment experiment(bounds, blocks);
    const Experiment planes(bounds, blocks, true);
    DeviceFused fused(experiment, false);
    DeviceExperiment current(planes, slabRows, false, MAX_MATCHES, true);
    float standardMs = 0;
    const auto expected = baselineSearch(bounds, blocks, {}, standardMs);
    if (reference && expected.total != 1)
        throw std::runtime_error("Full reference must return exactly one match");
    fused.run<MatchMode::Full>();
    const auto counts = fused.counts();
    compareResults(fused.results(), expected, planes);
    if (reference)
    {
        const Counts cpu{fusedCpuCosetZero(experiment, threads), 24417246, 24412805, 24406584, 390623920};
        if (counts != cpu)
            throw std::runtime_error("Full fused counts differ from CPU oracle");
    }
    Word survivors = 0;
    for (Word count : counts)
        survivors += count;
    const auto &layout = experiment.layout;
    const auto candidates = static_cast<Word>(layout.bounds.x_max - layout.bounds.x_min + 1) *
                            (layout.bounds.y_max - layout.bounds.y_min + 1) *
                            (layout.bounds.z_max - layout.bounds.z_min + 1);
    std::printf("FUSED_COUNTS %s matches=%llu survivors=%llu candidates=%llu cosets",
                label, expected.total, survivors, candidates);
    for (Word count : counts)
        std::printf(" %llu", count);
    std::putchar('\n');
    Sample warm{};
    fused.run<MatchMode::NoTail>();
    fused.run<MatchMode::Search>();
    current.run(warm, nullptr, MatchMode::NoTail);
    compareResults(current.run(warm, nullptr, MatchMode::Search), expected, planes);
    for (int i = 0; i < SAMPLES; ++i)
    {
        Sample filter{}, search{};
        float fusedFilter = 0, fusedTotal = 0;
        for (int j = 0; j < 2; ++j)
        {
            if ((i + j) % 2)
            {
                fusedFilter = fused.run<MatchMode::NoTail>();
                fusedTotal = fused.run<MatchMode::Search>();
                compareResults(fused.results(), expected, planes);
            }
            else
            {
                current.run(filter, nullptr, MatchMode::NoTail);
                compareResults(current.run(search, nullptr, MatchMode::Search), expected, planes);
            }
        }
        std::printf("FUSED_PAIR %s %d %.6f %.6f %.6f %.6f %.6f\n",
                    label, i + 1, filter.fill, filter.matchAndVerify,
                    search.fill + search.matchAndVerify, fusedFilter, fusedTotal);
    }
    std::printf("PASS fused %s exact matches in all timed samples\n", label);
}

void runFusedExperiment(const SearchBounds &bounds, const std::vector<BlockInfo> &blocks,
                        int slabRows, unsigned int threads)
{
    const SearchBounds tiny{-78, 67, -3, 3, -7, -4};
    std::vector<BlockInfo> synthetic{{-4, -2, -6, 0, false}, {1, 0, -6, 0, false},
                                     {6, -1, 2, 0, false}, {0, 0, 7, 0, false}};
    for (auto &b : synthetic)
        b.rotation = getTextureModern(-11 + b.x, -1 + b.y, -6 + b.z, 4);
    verifyFused(tiny, synthetic, nullptr, true);
    auto oneRow = tiny;
    oneRow.z_min = oneRow.z_max;
    verifyFused(oneRow, synthetic, nullptr, true);
    const SearchBounds slice{bounds.x_min, bounds.x_min + 19999, bounds.y_min, bounds.y_max,
                             bounds.z_min, bounds.z_min + 119};
    const Counts expected{46920, 11776, 11649, 11527, 188452};
    verifyFused(slice, blocks, &expected, true);
    std::puts("PASS fused synthetic and reference slice masks and exact matches; slice counts 46920 11776 11649 11527 188452");

    timeFused("reference", bounds, blocks, slabRows, threads, true);
    std::vector<BlockInfo> line;
    for (int z = 0; z < 12; ++z)
        line.push_back({0, 0, z, z % 4, false});
    SearchBounds lineBounds = bounds;
    lineBounds.y_max = lineBounds.y_min;
    timeFused("line", lineBounds, line, slabRows, threads, false);
}

void comparePacking(const SearchBounds &bounds, const std::vector<BlockInfo> &formation, int slabRows)
{
    const Experiment experiment(bounds, formation, true);
    DeviceExperiment original(experiment, slabRows, false);
    DeviceExperiment packed(experiment, slabRows, false);
    const auto check = [&](int offset) {
        const auto slab = original.slabAt(offset);
        const size_t count = product(product(slab.ySlots, slab.hashedRows), slab.fieldWords);
        original.fillSlab(slab);
        packed.fillSlab(slab, true);
        // Compare in chunks to keep host memory small even on a full slab.
        constexpr size_t chunk = 1 << 18;
        std::vector<Planes> a(chunk), b(chunk);
        for (size_t i = 0; i < count; i += chunk)
        {
            const auto n = std::min(chunk, count - i);
            checkCuda(cudaMemcpy(a.data(), original.field() + i, n * sizeof(Planes), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(b.data(), packed.field() + i, n * sizeof(Planes), cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < n; ++j)
                if (a[j].low != b[j].low || a[j].high != b[j].high)
                    throw std::runtime_error("32-bit packing changed a plane word");
        }
        return count;
    };
    std::printf("PACK_CHECK words=%zu\n", check(0));
    if (experiment.layout.nz <= slabRows)
        return;

    const auto timeFill = [&](bool pack32) {
        float ms = 0;
        auto &device = pack32 ? packed : original;
        for (int offset = 0; offset < experiment.layout.nz;)
        {
            const auto slab = device.slabAt(offset);
            ms += device.fillSlab(slab, pack32);
            offset += slab.nz;
        }
        return ms;
    };
    timeFill(false);
    timeFill(true);
    for (int i = 0; i < SAMPLES; ++i)
    {
        std::array<float, 2> times{};
        for (int j = 0; j < 2; ++j)
        {
            const bool pack32 = (i + j) % 2 != 0;
            times[pack32] = timeFill(pack32);
        }
        std::printf("PACK_PAIR %d %.6f %.6f\n", i + 1, times[0], times[1]);
    }
}

void comparePackingSearch(const SearchBounds &bounds, const std::vector<BlockInfo> &formation,
                          int slabRows)
{
    const Experiment experiment(bounds, formation, true);
    const auto side = std::find_if(formation.begin(), formation.end(),
                                   [](const BlockInfo &block) { return block.isSide; });
    float standardMs = 0;
    const auto expected = baselineSearch(bounds, {formation.begin(), side},
                                         {side, formation.end()}, standardMs);
    if (expected.total != 1)
        throw std::runtime_error("Expected one match on the packing reference workload");

    const auto run = [&](bool pack32) {
        Sample sample{};
        SearchResult result;
        const auto start = std::chrono::steady_clock::now();
        {
            DeviceExperiment device(experiment, slabRows, false, MAX_MATCHES, true);
            result = device.run(sample, nullptr, MatchMode::Search, pack32);
        }
        sample.prototypeWall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        compareResults(result, expected, experiment);
        return sample;
    };
    run(false);
    run(true);
    for (int i = 0; i < SAMPLES; ++i)
    {
        std::array<Sample, 2> pair{};
        for (int j = 0; j < 2; ++j)
        {
            const bool pack32 = (i + j) % 2 != 0;
            pair[pack32] = run(pack32);
        }
        std::printf("PACK_SEARCH %d %.6f %.6f %.6f %.6f %.6f %.6f\n", i + 1,
                    pair[0].fill, pair[1].fill,
                    pair[0].fill + pair[0].matchAndVerify, pair[1].fill + pair[1].matchAndVerify,
                    pair[0].prototypeWall, pair[1].prototypeWall);
    }
    std::puts("PASS packing search exact standard-kernel match in every run");
}

int parseInt(std::string_view value)
{
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::runtime_error("Expected an integer argument");
    return result;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 8 || argc > 10)
    {
        std::fprintf(stderr, "Usage: %s FIXTURE X_MIN X_MAX Y_MIN Y_MAX Z_MIN Z_MAX "
                     "[CPU_THREADS [SLAB_ROWS]]\n", argv[0]);
        return 1;
    }

    try
    {
        const SearchBounds bounds{parseInt(argv[2]), parseInt(argv[3]), parseInt(argv[4]),
                                  parseInt(argv[5]), parseInt(argv[6]), parseInt(argv[7])};
        const int threads = argc >= 9 ? parseInt(argv[8]) : 1;
        const int slabRows = argc == 10 ? parseInt(argv[9]) : DEFAULT_SLAB_ROWS;
        if (threads < 1 || slabRows < 1)
            throw std::runtime_error("CPU_THREADS and SLAB_ROWS must be positive");

        auto formation = parseFormationFile(argv[1]);
        const auto sidesBegin = std::stable_partition(formation.begin(), formation.end(),
            [](const BlockInfo &block) { return !block.isSide; });
        const std::vector<BlockInfo> tops(formation.begin(), sidesBegin);
        const std::vector<BlockInfo> sides(sidesBegin, formation.end());
        if (std::getenv("TF_FLAT_COMPARE_PACKING"))
        {
            const std::vector<BlockInfo> synthetic{{-4, -2, -3, 0, false}, {0, 0, 3, 0, false}};
            for (int width : {1, 31, 32, 33, 63, 64, 65, 127})
                comparePacking({-131, -131 + width - 1, -3, 2, -12, 8}, synthetic, 21);
            comparePacking(bounds, formation, slabRows);
            return 0;
        }
        if (std::getenv("TF_FLAT_COMPARE_PACKING_SEARCH"))
        {
            comparePackingSearch(bounds, formation, slabRows);
            return 0;
        }
        if (std::getenv("TF_FLAT_FUSED"))
        {
            runFusedExperiment(bounds, formation, slabRows, threads);
            return 0;
        }
        if (std::getenv("TF_FLAT_SHARED_FIELD"))
        {
            const SearchBounds small{-70, 60, -2, 1, -7, 4};
            std::vector<BlockInfo> synthetic{{-77, -1, -3, 0, false}, {18, 2, 67, 0, false},
                                             {2, 0, 4, 0, false}, {66, 1, -1, 0, true}};
            for (auto &block : synthetic)
                block.rotation = getTextureModern(1 + block.x, block.y, -2 + block.z,
                                                  block.isSide ? MOD_SIDE : MOD_TOP_BOTTOM);
            compareSharedField(small, synthetic, 1, true);
            for (int dir = 1; dir < 4; ++dir)
            {
                for (auto &block : synthetic)
                    rotateFormation(block);
                compareSharedField(small, synthetic, 5, true);
            }
            std::puts("PASS shared masks match independent GPU fields and CPU on synthetic slabs");
            compareSharedField(bounds, formation, slabRows, false);
            return 0;
        }
        const bool worldY = std::getenv("TF_FLAT_DUPLICATE_Y") == nullptr;
        const Experiment experiment(bounds, formation, worldY);
        if (std::getenv("TF_FLAT_COMPARE_OFFSETS"))
        {
            float baselineMs = 0;
            const auto reference = baselineSearch(bounds, tops, sides, baselineMs);
            comparePlaneOffsets(experiment, slabRows, reference);
            return 0;
        }
        verifySmallCase(worldY, 4, false);
        if (worldY)
        {
            verifySmallCase(worldY, 3, true);
            verifySmallCase(worldY, 1, true);
        }
        using Clock = std::chrono::steady_clock;
        using Milliseconds = std::chrono::duration<double, std::milli>;
        const auto baseline = [&](Sample &sample) {
            const auto start = Clock::now();
            auto result = baselineSearch(bounds, tops, sides, sample.baseline);
            sample.baselineWall = Milliseconds(Clock::now() - start).count();
            return result;
        };
        const auto prototype = [&](Sample &sample) {
            const auto start = Clock::now();
            SearchResult result;
            {
                DeviceExperiment device(experiment, slabRows, false);
                result = device.run(sample);
            }
            sample.prototypeWall = Milliseconds(Clock::now() - start).count();
            return result;
        };
        Sample warmup{};
        const auto reference = baseline(warmup);
        Counts counts;
        {
            DeviceExperiment device(experiment, slabRows, false);
            std::printf("SEARCH candidates=%zu plane_bytes=%zu slabs=%d\n",
                        product(experiment.rows(), experiment.layout.nx), device.planeBytes(),
                        device.slabCount());
            compareResults(device.run(warmup, &counts), reference, experiment);
        }

        std::array<Sample, SAMPLES> samples;
        for (int i = 0; i < SAMPLES; ++i)
        {
            auto &sample = samples[i];
            SearchResult actual, expected;
            if (i % 2 == 0)
            {
                expected = baseline(sample);
                actual = prototype(sample);
            }
            else
            {
                actual = prototype(sample);
                expected = baseline(sample);
            }
            compareResults(actual, reference, experiment);
            compareResults(expected, reference, experiment);
        }
        const Counts expected = referenceCounts(experiment, threads);
        if (counts != expected)
            throw std::runtime_error("GPU/CPU survivor counts differ on the search volume");
        for (int coset = 0; coset < STRIDE; ++coset)
            std::printf("COSET %d %llu\n", coset, expected[coset]);
        std::printf("PASS full GPU/CPU survivor counts and baseline matches (total=%llu)\n", reference.total);
        if (const char *diagnostics = std::getenv("TF_FLAT_DIAGNOSTICS"))
            runDiagnostics(experiment, slabRows, expected, std::string_view(diagnostics) == "all");
        for (int i = 0; i < SAMPLES; ++i)
            std::printf("SAMPLE %d baseline_ms=%.6f fill_ms=%.6f match_verify_ms=%.6f "
                        "baseline_wall_ms=%.6f prototype_wall_ms=%.6f\n",
                        i + 1, samples[i].baseline, samples[i].fill, samples[i].matchAndVerify,
                        samples[i].baselineWall, samples[i].prototypeWall);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
