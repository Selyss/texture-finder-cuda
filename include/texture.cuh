#pragma once
#include <cstdint>

// Host- and device-compilable so the exact same code is used by the kernel,
// the test suite, and the formation generator. Never copy these functions.
#ifdef __CUDACC__
#define TF_HOST_DEVICE __host__ __device__
#else
#define TF_HOST_DEVICE
#endif

// Reference semantics: 19MisterX98/TextureRotations (texture/*.java) and the
// vanilla Mth.getSeed position hash. Java overflow wraps; C++ signed overflow
// is UB, so every wrapping step below is routed through unsigned arithmetic.

// Java: long l = (long)(x * 3129871) ^ (long)z * 116129781L ^ (long)y;
//       l = l * l * 42317861L + l * 11L;
//       return l >> 16;
// x * 3129871 is 32-bit Java int math (wraps); the rest is 64-bit.
TF_HOST_DEVICE inline int64_t coordRandom(int32_t x, int32_t y, int32_t z)
{
    int64_t l = (int64_t)(int32_t)((uint32_t)x * 3129871u)
              ^ ((int64_t)z * 116129781LL)
              ^ (int64_t)y;
    uint64_t ul = (uint64_t)l;
    ul = ul * ul * 42317861ull + ul * 11ull;
    return (int64_t)ul >> 16;
}

// Versions 1.13 - 1.21.1 (reference: Vanilla21_1Textures).
// Java: seed = (coordRandom ^ 0x5DEECE66D) & ((1<<48)-1);
//       rand = (int)((seed * 0xBB20B4600A69 + 0x40942DE6BA) >>> 16);  // nextLong, 2 LCG steps combined
//       return Math.abs(rand) % mod;
// The unsigned-negate trick equals Java Math.abs for mod 2 and 4 even at
// rand == INT_MIN (both yield 0), without the UB of abs(INT_MIN).
TF_HOST_DEVICE inline int getTextureLegacy(int32_t x, int32_t y, int32_t z, int mod)
{
    constexpr int64_t MULTIPLIER = 0x5DEECE66DLL;
    constexpr int64_t MASK = (1LL << 48) - 1;
    int64_t seed = (coordRandom(x, y, z) ^ MULTIPLIER) & MASK;
    uint64_t v = (uint64_t)seed * 0xBB20B4600A69ull + 0x40942DE6BAull;
    int32_t rand = (int32_t)(uint32_t)(v >> 16);
    uint32_t a = rand < 0 ? 0u - (uint32_t)rand : (uint32_t)rand;
    return (int)(a % (uint32_t)mod);
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
TF_HOST_DEVICE inline int getTextureModern(int32_t x, int32_t y, int32_t z, int mod)
{
    constexpr int64_t MULTIPLIER = 0x5DEECE66DLL;
    constexpr int64_t MASK = (1LL << 48) - 1;
    int64_t seed = coordRandom(x, y, z) ^ MULTIPLIER;
    seed = (int64_t)((uint64_t)seed * (uint64_t)MULTIPLIER + 11ull) & MASK;
    int next = (int)(seed >> (48 - 31));
    return (int)(((4LL * next) >> 31) % mod);
}
