
// TODO: implement the other versions of the texture rotation generation

/**
 * @brief Mixes the input seed using Stafford's mix13 algorithm.
 *
 * @param z The input seed value.
 * @return A pseudo-random integer derived from the input seed.
 */
__device__ int staffordMix13(long z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9L;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBL;
    return static_cast<int>(z ^ (z >> 31));
}

/**
 * @brief Legacy texture rotation generator for versions 1.13 - 1.21.1.
 *
 * @param x The x-coordinate
 * @param y The y-coordinate
 * @param z The z-coordinate
 * @param mod The number of rotation states
 * @return A pseudo-random rotation state in the range [0, mod).
 */
__device__ int getTextureLegacy(int x, int y, int z, int mod)
{
    constexpr int64_t multiplier = 0x5DEECE66DLL;
    constexpr int64_t mask = (1LL << 48) - 1;
    int64_t l = (static_cast<int64_t>(x * 3129871)) ^
                (static_cast<int64_t>(z) * 116129781L) ^ static_cast<int64_t>(y);
    l = l * l * 42317861L + l * 11L;
    int64_t seed = l >> 16;
    seed = (seed ^ multiplier) & mask;
    int rand = static_cast<int>((seed * 0xBB20B4600A69L + 0x40942DE6BAL) >> 16);
    return abs(rand) % mod;
}

/**
 * @brief Modern texture rotation generator for versions 1.21.2 and later (at the time of writing).
 *
 * @param x The x-coordinate
 * @param y The y-coordinate
 * @param z The z-coordinate
 * @param mod The number of rotation states
 * @return A pseudo-random rotation state in the range [0, mod).
 */
__device__ int getTextureModern(int x, int y, int z, int mod)
{
    constexpr int64_t multiplier = 0x5DEECE66DLL;
    constexpr int64_t mask = (1LL << 48) - 1;
    int64_t l = (static_cast<int64_t>(x * 3129871)) ^
                (static_cast<int64_t>(z) * 116129781L) ^ static_cast<int64_t>(y);
    l = l * l * 42317861L + l * 11L;
    int64_t seed = l >> 16;
    seed = (seed ^ multiplier);
    seed = seed * multiplier + 11LL & mask;
    int next = static_cast<int>(seed >> (48 - 31));
    return static_cast<int>((mod * static_cast<int64_t>(next)) >> 31);
}