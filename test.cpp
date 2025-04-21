#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <sstream>

int getTextureModern(int x, int y, int z, int mod)
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

int getTextureLegacy(int x, int y, int z, int mod)
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

TEST_CASE("getTextureModern")
{
    CHECK(getTextureModern(-1, 0, -1, 4) == 0);
    CHECK(getTextureModern(0, 0, -1, 4) == 3);
    CHECK(getTextureModern(1, 0, -1, 4) == 3);
    CHECK(getTextureModern(-2, 0, 0, 4) == 3);
    CHECK(getTextureModern(-1, 0, 0, 4) == 0);
    CHECK(getTextureModern(0, 0, 0, 4) == 2);
    CHECK(getTextureModern(1, 0, 0, 4) == 0);
    CHECK(getTextureModern(-1, 0, 1, 4) == 3);
    CHECK(getTextureModern(0, 0, 1, 4) == 1);
    CHECK(getTextureModern(1, 0, 1, 4) == 0);
}

TEST_CASE("getTextureLegacy - mod 4")
{
    CHECK(getTextureLegacy(-1, 0, -1, 4) == 0);
    CHECK(getTextureLegacy(0, 0, -1, 4) == 1);
    CHECK(getTextureLegacy(1, 0, -1, 4) == 0);
    CHECK(getTextureLegacy(-2, 0, 0, 4) == 2);
    CHECK(getTextureLegacy(-1, 0, 0, 4) == 0);
    CHECK(getTextureLegacy(0, 0, 0, 4) == 0);
    CHECK(getTextureLegacy(1, 0, 0, 4) == 1);
    CHECK(getTextureLegacy(-1, 0, 1, 4) == 0);
    CHECK(getTextureLegacy(0, 0, 1, 4) == 3);
    CHECK(getTextureLegacy(1, 0, 1, 4) == 0);
}

TEST_CASE("getTextureLegacy - mod 2")
{
    CHECK(getTextureLegacy(-6, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(-5, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(-4, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(-3, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(-2, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(-1, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(0, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(1, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(2, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(3, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(4, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(5, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(6, 0, 0, 2) == 0);
    CHECK(getTextureLegacy(7, 0, 0, 2) == 1);
    CHECK(getTextureLegacy(8, 0, 0, 2) == 0);
}