#pragma once

// Compilable as C++, CUDA, and Metal Shading Language so the exact same
// code is used by every backend, the test suite, and the tools. Never copy
// these functions.
#ifdef __METAL_VERSION__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;           // MSL long/ulong are 64-bit
typedef unsigned long uint64_t;
#define TF_HOST_DEVICE
#else
#include <cstdint>
#ifdef __CUDACC__
#define TF_HOST_DEVICE __host__ __device__
#else
#define TF_HOST_DEVICE
#endif
#endif

// Reference semantics: 19MisterX98/TextureRotations (texture/*.java) and the
// vanilla Mth.getSeed position hash. Java overflow wraps; C++ signed overflow
// is UB, so every wrapping step below is routed through unsigned arithmetic.

// Java: long l = (long)(x * 3129871) ^ (long)z * 116129781L ^ (long)y;
//       l = l * l * 42317861L + l * 11L;
//       return l >> 16;
// x * 3129871 is 32-bit Java int math (wraps); the rest is 64-bit.
//
// The three coordinate terms are exposed separately (coordMixFromParts)
// because both multiplications distribute over the formation offsets under
// wrapping: (x+bx)*3129871 mod 2^32 == x*3129871 + bx*3129871 mod 2^32, and
// the z product is exact in 64 bits. The kernel exploits this to hoist the
// per-candidate multiplies out of the per-block loop; the fused coordRandom
// below stays the reference form.
//
// coordMix is the full 64-bit mixed value BEFORE the >> 16: most versions
// consume mix >> 16 (reference getCoordinateRandom), but <=1.12 truncates
// the full value to int first (reference getCoordinateRandomLegacy).
TF_HOST_DEVICE inline int64_t coordMixFromParts(uint32_t xTerm, int64_t zTerm, int32_t yTerm)
{
    int64_t l = (int64_t)(int32_t)xTerm ^ zTerm ^ (int64_t)yTerm;
    uint64_t ul = (uint64_t)l;
    ul = ul * ul * 42317861ull + ul * 11ull;
    return (int64_t)ul;
}

TF_HOST_DEVICE inline int64_t coordRandomFromParts(uint32_t xTerm, int64_t zTerm, int32_t yTerm)
{
    return coordMixFromParts(xTerm, zTerm, yTerm) >> 16;
}

TF_HOST_DEVICE inline int64_t coordMix(int32_t x, int32_t y, int32_t z)
{
    return coordMixFromParts((uint32_t)x * 3129871u, (int64_t)z * 116129781LL, y);
}

TF_HOST_DEVICE inline int64_t coordRandom(int32_t x, int32_t y, int32_t z)
{
    return coordMix(x, y, z) >> 16;
}

// Java Math.abs(rand) % mod, computed in unsigned arithmetic: identical for
// mod 2 and 4 even at rand == INT_MIN (both yield 0), and never UB.
TF_HOST_DEVICE inline int absMod(int32_t rand, int mod)
{
    uint32_t a = rand < 0 ? 0u - (uint32_t)rand : (uint32_t)rand;
    return (int)(a % (uint32_t)mod);
}

// Versions 1.13 - 1.21.1 (reference: Vanilla21_1Textures).
// Java: seed = (coordRandom ^ 0x5DEECE66D) & ((1<<48)-1);
//       rand = (int)((seed * 0xBB20B4600A69 + 0x40942DE6BA) >>> 16);  // nextLong, 2 LCG steps combined
//       return Math.abs(rand) % mod;
// The unsigned-negate trick equals Java Math.abs for mod 2 and 4 even at
// rand == INT_MIN (both yield 0), without the UB of abs(INT_MIN).
TF_HOST_DEVICE inline int legacyFromCoordRandom(int64_t coordRand, int mod)
{
    constexpr int64_t MULTIPLIER = 0x5DEECE66DLL;
    constexpr int64_t MASK = (1LL << 48) - 1;
    int64_t seed = (coordRand ^ MULTIPLIER) & MASK;
    uint64_t v = (uint64_t)seed * 0xBB20B4600A69ull + 0x40942DE6BAull;
    return absMod((int32_t)(uint32_t)(v >> 16), mod);
}

TF_HOST_DEVICE inline int getTextureLegacy(int32_t x, int32_t y, int32_t z, int mod)
{
    return legacyFromCoordRandom(coordRandom(x, y, z), mod);
}

// Versions <= 1.12.2 (reference: Vanilla12Textures).
// Java: rand = (int) coordMix >> 16;   // truncate to int FIRST, then shift
//       return Math.abs(rand) % mod;
// No LCG scramble at all in this era. The int64 detour below performs the
// arithmetic >> 16 with fully defined semantics.
TF_HOST_DEVICE inline int vanilla12FromMix(int64_t mix, int mod)
{
    int32_t truncated = (int32_t)(uint32_t)(uint64_t)mix;
    int32_t rand = (int32_t)((int64_t)truncated >> 16);
    return absMod(rand, mod);
}

TF_HOST_DEVICE inline int getTextureVanilla12(int32_t x, int32_t y, int32_t z, int mod)
{
    return vanilla12FromMix(coordMix(x, y, z), mod);
}

// Java: (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9; (z ^ (z >>> 27)) * 0x94D049BB133111EB; z ^ (z >>> 31)
TF_HOST_DEVICE inline uint64_t staffordMix13(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Sodium 4.2 - 4.8 on MC 1.19 - 1.19.3 (reference: Sodium19Textures).
// Java: l = seed ^ 7640891576956012809L; m = l + -7046029254386353131L;
//       rand = (int)(Long.rotateLeft(mix13(l) + mix13(m), 17) + mix13(l));
//       return Math.abs(rand) % mod;
// (-7046029254386353131 as u64 is 0x9E3779B97F4A7C15, the golden ratio.)
TF_HOST_DEVICE inline int sodium19FromCoordRandom(int64_t coordRand, int mod)
{
    uint64_t l = (uint64_t)coordRand ^ 0x6A09E667F3BCC909ull;
    uint64_t m = l + 0x9E3779B97F4A7C15ull;
    l = staffordMix13(l);
    m = staffordMix13(m);
    uint64_t sum = l + m;
    uint64_t rot = (sum << 17) | (sum >> 47); // Long.rotateLeft(sum, 17)
    return absMod((int32_t)(uint32_t)(rot + l), mod);
}

TF_HOST_DEVICE inline int getTextureSodium19(int32_t x, int32_t y, int32_t z, int mod)
{
    return sodium19FromCoordRandom(coordRandom(x, y, z), mod);
}

// Sodium 1.0 - 4.1 on MC 1.16 - 1.18.2 (reference: SodiumTextures).
// Java: murmur-style avalanche of the seed, then
//       rand1 = mix13(seed += PHI); rand2 = mix13(seed + PHI);
//       rand = (int)(rand1 + rand2); return Math.abs(rand) % mod;
TF_HOST_DEVICE inline int sodiumFromCoordRandom(int64_t coordRand, int mod)
{
    uint64_t s = (uint64_t)coordRand;
    s ^= s >> 33;
    s *= 0xFF51AFD7ED558CCDull;
    s ^= s >> 33;
    s *= 0xC4CEB9FE1A85EC53ull;
    s ^= s >> 33;
    s += 0x9E3779B97F4A7C15ull;                          // seed += PHI
    uint64_t rand1 = staffordMix13(s);
    uint64_t rand2 = staffordMix13(s + 0x9E3779B97F4A7C15ull); // seed + PHI again
    return absMod((int32_t)(uint32_t)(rand1 + rand2), mod);
}

TF_HOST_DEVICE inline int getTextureSodium(int32_t x, int32_t y, int32_t z, int mod)
{
    return sodiumFromCoordRandom(coordRandom(x, y, z), mod);
}

// Versions 1.21.2+ (reference: VanillaTextures).
// Java: seed = coordRandom ^ 0x5DEECE66D;
//       seed = seed * 0x5DEECE66D + 11 & ((1<<48)-1);
//       next = (int)(seed >> 48 - 31);                       // nextInt(4) bound path
//       return (int)((4 * (long)next) >> 31) % mod;
// The game always rolls one of 4 model variants; a side face only exposes the
// parity of that roll, hence the hardcoded 4 with % mod applied after. Using
// nextInt(2)-style (mod * next) >> 31 for sides is WRONG - it draws a number
// the game never draws.
TF_HOST_DEVICE inline int modernFromCoordRandom(int64_t coordRand, int mod)
{
    constexpr int64_t MULTIPLIER = 0x5DEECE66DLL;
    constexpr int64_t MASK = (1LL << 48) - 1;
    int64_t seed = coordRand ^ MULTIPLIER;
    seed = (int64_t)((uint64_t)seed * (uint64_t)MULTIPLIER + 11ull) & MASK;
    // seed is in [0, 2^48) after the mask, so `next` is non-negative and the
    // Java tail is computed in unsigned form (identical values for all
    // inputs; this lets the compiler fold the shift and constant mod instead
    // of emitting signed-division fixups it can't prove away).
    unsigned int next = (unsigned int)((uint64_t)seed >> (48 - 31));
    return (int)(((4ull * next) >> 31) % (unsigned int)mod);
}

TF_HOST_DEVICE inline int getTextureModern(int32_t x, int32_t y, int32_t z, int mod)
{
    return modernFromCoordRandom(coordRandom(x, y, z), mod);
}

// Runtime dispatch over the version constants in blockinfo.cuh (host tools
// and the CPU backend; the GPU kernels dispatch at compile time instead).
// 0 = modern (1.21.2+), 1 = legacy (1.13-1.21.1), 2 = vanilla <=1.12.2,
// 3 = Sodium 1.16-1.18.2, 4 = Sodium 1.19-1.19.3.
TF_HOST_DEVICE inline int getTextureForVersion(int version, int32_t x, int32_t y, int32_t z, int mod)
{
    switch (version)
    {
    case 1:
        return getTextureLegacy(x, y, z, mod);
    case 2:
        return getTextureVanilla12(x, y, z, mod);
    case 3:
        return getTextureSodium(x, y, z, mod);
    case 4:
        return getTextureSodium19(x, y, z, mod);
    default:
        return getTextureModern(x, y, z, mod);
    }
}
