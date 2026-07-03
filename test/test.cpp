#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// The real headers the kernel compiles — not copies.
#include "blockinfo.cuh"
#include "parser.cuh"
#include "rotation.cuh"
#include "texture.cuh"

// Verified in-game vectors for versions 1.21.2+ (top faces, mod 4).
TEST_CASE("getTextureModern - top faces")
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

// The game rolls one of 4 model variants per position; a side face exposes
// only the parity of that roll. Reference: VanillaTextures.getTexture (the
// hardcoded 4) and RotationInfo (rotation % 2 for sides).
TEST_CASE("getTextureModern - side faces are the parity of the top roll")
{
    CHECK(getTextureModern(0, 0, 0, 2) == 0);
    CHECK(getTextureModern(1, 0, 0, 2) == 0);
    CHECK(getTextureModern(-1, 0, 0, 2) == 0);
    CHECK(getTextureModern(-2, 0, 0, 2) == 1);
    CHECK(getTextureModern(0, 0, 1, 2) == 1);
    CHECK(getTextureModern(0, 0, -1, 2) == 1);

    for (int x = -64; x <= 64; x++)
        for (int z = -64; z <= 64; z++)
            for (int y : {-64, 0, 200})
            {
                CHECK(getTextureModern(x, y, z, 2) == getTextureModern(x, y, z, 4) % 2);
                CHECK(getTextureLegacy(x, y, z, 2) == getTextureLegacy(x, y, z, 4) % 2);
            }
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

// Real-world verified formation: see test/fixtures/formation_a.expected.md.
TEST_CASE("formation_a matches at its verified origin")
{
    const int OX = -108723, OY = -54, OZ = -69736;
    const int blocks[24][4] = {
        {1, 0, 0, 0}, {2, 0, 0, 2}, {0, 0, 1, 2}, {1, 0, 1, 3}, {2, 0, 1, 3},
        {0, 0, 2, 3}, {1, 0, 2, 3}, {2, 0, 2, 0}, {3, 2, 2, 1}, {4, 2, 2, 2},
        {0, 0, 3, 1}, {1, 0, 3, 2}, {2, 0, 3, 1}, {3, 2, 3, 2}, {4, 2, 3, 1},
        {0, 0, 4, 1}, {1, 0, 4, 1}, {2, 0, 4, 0}, {3, 2, 4, 2}, {4, 2, 4, 0},
        {0, 0, 5, 3}, {1, 0, 5, 3}, {2, 0, 5, 0}, {3, 2, 5, 3}};
    for (const auto &b : blocks)
        CHECK(getTextureModern(OX + b[0], OY + b[1], OZ + b[2], 4) == b[3]);
}

TEST_CASE("rotateFormation - coordinates cycle east -> south -> west -> north")
{
    BlockInfo b{1, 5, 0, 0, false};
    rotateFormation(b);
    CHECK(b.x == 0);
    CHECK(b.z == 1);
    rotateFormation(b);
    CHECK(b.x == -1);
    CHECK(b.z == 0);
    rotateFormation(b);
    CHECK(b.x == 0);
    CHECK(b.z == -1);
    rotateFormation(b);
    CHECK(b.x == 1);
    CHECK(b.z == 0);
    CHECK(b.y == 5);
}

TEST_CASE("rotateFormation - top rotations advance mod 4, side rotations flip parity")
{
    BlockInfo top{2, 0, 3, 3, false};
    rotateFormation(top);
    CHECK(top.rotation == 0);

    BlockInfo side{2, 0, 3, 1, true};
    rotateFormation(side);
    CHECK(side.rotation == 0);
    rotateFormation(side);
    CHECK(side.rotation == 1);

    // A side value must stay in {0, 1} after any number of rotations; the
    // old code let it reach 2-3, which a mod-2 texture can never equal.
    BlockInfo s{0, 0, 0, 1, true};
    for (int i = 0; i < 7; i++)
    {
        rotateFormation(s);
        CHECK(s.rotation >= 0);
        CHECK(s.rotation <= 1);
    }
}

TEST_CASE("rotateFormation - four rotations are the identity")
{
    std::vector<BlockInfo> formation = {
        {1, 0, 0, 0, false}, {2, 1, -3, 3, false}, {-4, 2, 5, 1, true}, {0, -1, 7, 0, true}};
    std::vector<BlockInfo> original = formation;
    for (int i = 0; i < 4; i++)
        for (auto &b : formation)
            rotateFormation(b);
    for (size_t i = 0; i < formation.size(); i++)
    {
        CHECK(formation[i].x == original[i].x);
        CHECK(formation[i].y == original[i].y);
        CHECK(formation[i].z == original[i].z);
        CHECK(formation[i].rotation == original[i].rotation);
    }
}

namespace
{
std::string writeTempFile(const std::string &content)
{
    static int counter = 0;
    std::string path = "build/parser_test_" + std::to_string(counter++) + ".txt";
    std::ofstream out(path);
    out << content;
    return path;
}
} // namespace

TEST_CASE("parser - valid file with comments and blank lines")
{
    std::string path = writeTempFile("# header comment\n1 2 3 2 0\n\n4 5 6 3 1\n");
    auto formation = parseFormationFile(path);
    REQUIRE(formation.size() == 2);
    CHECK(formation[0].x == 1);
    CHECK(formation[0].rotation == 2);
    CHECK(formation[0].isSide == false);
    CHECK(formation[1].isSide == true);
    // Side rotations are reduced to parity at parse time (reference: RotationInfo).
    CHECK(formation[1].rotation == 1);
    std::remove(path.c_str());
}

TEST_CASE("parser - rejects bad input loudly instead of skipping it")
{
    CHECK_THROWS_AS(parseFormationFile("does_not_exist_12345.txt"), std::runtime_error);
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("")), std::runtime_error);
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("# only a comment\n")), std::runtime_error);
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("1 2 3 4 0\n")), std::runtime_error);  // rotation > 3
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("1 2 3 -1 0\n")), std::runtime_error); // rotation < 0
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("1 2 3 1 2\n")), std::runtime_error);  // isSide not 0/1
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("1 2 3\n")), std::runtime_error);      // missing fields
    CHECK_THROWS_AS(parseFormationFile(writeTempFile("a b c d e\n")), std::runtime_error);
}
