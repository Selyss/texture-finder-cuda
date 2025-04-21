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