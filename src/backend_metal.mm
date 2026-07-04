// Apple GPU backend via Metal. The compute shader is built at runtime from
// a source string generated at build time by concatenating
// include/texture.cuh and src/match.metal (see the Makefile), so the RNG is
// the same single source every other backend compiles. Work is dispatched
// in z-slices to keep individual command buffers short.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernel.cuh"

const char *searchBackendName() { return "Metal"; }

namespace
{

const char kShaderBody[] =
#include "tf_shader_src.h"
    ;

struct TfParams
{
    int x_min, x_max;
    int y_min, y_max;
    int z_max;
    int zBase;
    int totalBlocks;
    int pad;
};

struct TfBlock
{
    int x, y, z, w;
};

[[noreturn]] void fail(const char *what, NSError *err)
{
    std::fprintf(stderr, "Metal error: %s%s%s\n", what,
                 err ? ": " : "",
                 err ? err.localizedDescription.UTF8String : "");
    std::exit(2);
}

} // namespace

namespace
{

// Pipeline cache: runSearch is called once per slice by the host-side
// progress/checkpoint driver, and shader compilation costs ~100 ms — done
// once per version instead of once per call.
id<MTLDevice> sDevice;
id<MTLCommandQueue> sQueue;
id<MTLComputePipelineState> sPSO[NUM_VERSIONS];

id<MTLComputePipelineState> ensurePipeline(int version)
{
    if (sPSO[version])
        return sPSO[version];

    if (!sDevice)
    {
        sDevice = MTLCreateSystemDefaultDevice();
        if (!sDevice)
        {
            std::fprintf(stderr,
                         "No Metal device available; rebuild with `make BACKEND=cpu`\n");
            std::exit(2);
        }
        sQueue = [sDevice newCommandQueue];
    }

    std::string src = "#define TF_MAX_MATCHES " + std::to_string(MAX_MATCHES) + "u\n";
    src += kShaderBody;

    NSError *err = nil;
    id<MTLLibrary> lib = [sDevice newLibraryWithSource:@(src.c_str())
                                               options:nil
                                                 error:&err];
    if (!lib)
        fail("shader compilation failed", err);

    MTLFunctionConstantValues *consts = [MTLFunctionConstantValues new];
    int versionConst = version;
    [consts setConstantValue:&versionConst type:MTLDataTypeInt atIndex:0];
    id<MTLFunction> fn = [lib newFunctionWithName:@"matchFormation"
                                   constantValues:consts
                                            error:&err];
    if (!fn)
        fail("function specialization failed", err);
    id<MTLComputePipelineState> pso = [sDevice newComputePipelineStateWithFunction:fn
                                                                             error:&err];
    if (!pso)
        fail("pipeline creation failed", err);
    sPSO[version] = pso;
    return pso;
}

} // namespace

std::vector<MatchResult> runSearch(const SearchBounds &bounds,
                                   const std::vector<BlockInfo> &topsAndBottoms,
                                   const std::vector<BlockInfo> &sides,
                                   int version,
                                   float *kernelMs,
                                   bool *truncated,
                                   unsigned long long *totalMatches)
{
    if (topsAndBottoms.size() > MAX_FORMATION_BLOCKS || sides.size() > MAX_FORMATION_BLOCKS)
    {
        std::fprintf(stderr, "Formation exceeds %d blocks per face type\n", MAX_FORMATION_BLOCKS);
        std::exit(1);
    }

    std::vector<TfBlock> blocks;
    blocks.reserve(topsAndBottoms.size() + sides.size());
    for (const auto &t : topsAndBottoms)
        blocks.push_back({t.x, t.y, t.z, t.rotation | (3 << 8)});
    for (const auto &s : sides)
        blocks.push_back({s.x, s.y, s.z, s.rotation | (1 << 8)});
    if (blocks.empty())
    {
        std::fprintf(stderr, "Empty formation\n");
        std::exit(1);
    }

    const unsigned long long nx = (unsigned long long)((long long)bounds.x_max - bounds.x_min + 1);
    const unsigned long long ny = (unsigned long long)((long long)bounds.y_max - bounds.y_min + 1);
    const unsigned long long nz = (unsigned long long)((long long)bounds.z_max - bounds.z_min + 1);
    if (ny > (~0ull) / nz || nx > (~0ull) / (ny * nz))
    {
        std::fprintf(stderr, "Search volume too large\n");
        std::exit(1);
    }

    std::vector<MatchResult> matches;
    unsigned long long total = 0;
    float ms = 0.f;

    @autoreleasepool
    {
        id<MTLComputePipelineState> pso = ensurePipeline(version);
        id<MTLDevice> device = sDevice;
        id<MTLCommandQueue> queue = sQueue;

        id<MTLBuffer> bBlocks = [device newBufferWithBytes:blocks.data()
                                                    length:blocks.size() * sizeof(TfBlock)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOut = [device newBufferWithLength:MAX_MATCHES * sizeof(MatchResult)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bWriteIdx = [device newBufferWithLength:sizeof(unsigned int)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bSlice = [device newBufferWithLength:sizeof(unsigned int)
                                                   options:MTLResourceStorageModeShared];
        *(unsigned int *)bWriteIdx.contents = 0;

        // Slice depth chosen so one dispatch stays around a billion threads.
        const unsigned long long perLayer = nx * ny;
        unsigned long long zStep = perLayer > 0 ? (1ull << 30) / perLayer : nz;
        if (zStep == 0)
            zStep = 1;
        if (zStep > nz)
            zStep = nz;

        const NSUInteger tew = pso.threadExecutionWidth;
        const NSUInteger maxTg = pso.maxTotalThreadsPerThreadgroup;
        const MTLSize tg = MTLSizeMake(tew, 1, std::max<NSUInteger>(1, std::min<NSUInteger>(256, maxTg) / tew));

        for (unsigned long long z0 = 0; z0 < nz; z0 += zStep)
        {
            const unsigned long long zCount = std::min(zStep, nz - z0);
            TfParams P = {bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max,
                          bounds.z_max, (int)(bounds.z_min + (long long)z0),
                          (int)blocks.size(), 0};
            *(unsigned int *)bSlice.contents = 0;

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bBlocks offset:0 atIndex:0];
            [enc setBytes:&P length:sizeof(P) atIndex:1];
            [enc setBuffer:bOut offset:0 atIndex:2];
            [enc setBuffer:bWriteIdx offset:0 atIndex:3];
            [enc setBuffer:bSlice offset:0 atIndex:4];
            const MTLSize grid = MTLSizeMake((nx + tg.width - 1) / tg.width,
                                             (ny + tg.height - 1) / tg.height,
                                             (zCount + tg.depth - 1) / tg.depth);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error)
                fail("command buffer failed", cb.error);
            ms += (float)((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
            total += *(unsigned int *)bSlice.contents;
        }

        const unsigned int wrote = *(unsigned int *)bWriteIdx.contents;
        const unsigned int stored = std::min(wrote, MAX_MATCHES);
        matches.assign((MatchResult *)bOut.contents, (MatchResult *)bOut.contents + stored);
    }

    if (kernelMs)
        *kernelMs = ms;
    if (truncated)
        *truncated = total > MAX_MATCHES;
    if (totalMatches)
        *totalMatches = total;

    std::sort(matches.begin(), matches.end(), [](const MatchResult &a, const MatchResult &b)
              { return a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z); });
    return matches;
}
