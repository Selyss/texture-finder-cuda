#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The real headers the kernel compiles — not copies.
#include "backends/cpu/row_filter.hpp"
#include "bitplanes/match.hpp"
#include "bitplanes/plan.hpp"
#include "texture/formation.hpp"
#include "texture/formation_io.hpp"
#include "texture/rotation.hpp"
#include "texture/texture_rules.hpp"

// Verified in-game vectors for versions 1.21.2+ (top faces, mod 4).
TEST_CASE("get_texture_modern - top faces") {
  CHECK(get_texture_modern(-1, 0, -1, 4) == 0);
  CHECK(get_texture_modern(0, 0, -1, 4) == 3);
  CHECK(get_texture_modern(1, 0, -1, 4) == 3);
  CHECK(get_texture_modern(-2, 0, 0, 4) == 3);
  CHECK(get_texture_modern(-1, 0, 0, 4) == 0);
  CHECK(get_texture_modern(0, 0, 0, 4) == 2);
  CHECK(get_texture_modern(1, 0, 0, 4) == 0);
  CHECK(get_texture_modern(-1, 0, 1, 4) == 3);
  CHECK(get_texture_modern(0, 0, 1, 4) == 1);
  CHECK(get_texture_modern(1, 0, 1, 4) == 0);
}

// The game rolls one of 4 model variants per position; a side face exposes
// only the parity of that roll. Reference: VanillaTextures.get_texture (the
// hardcoded 4) and RotationInfo (rotation % 2 for sides).
TEST_CASE("get_texture_modern - side faces are the parity of the top roll") {
  CHECK(get_texture_modern(0, 0, 0, 2) == 0);
  CHECK(get_texture_modern(1, 0, 0, 2) == 0);
  CHECK(get_texture_modern(-1, 0, 0, 2) == 0);
  CHECK(get_texture_modern(-2, 0, 0, 2) == 1);
  CHECK(get_texture_modern(0, 0, 1, 2) == 1);
  CHECK(get_texture_modern(0, 0, -1, 2) == 1);

  for (int x = -64; x <= 64; x++)
    for (int z = -64; z <= 64; z++)
      for (int y : {-64, 0, 200})
        for (int index = 0; index < NUM_VERSIONS; index++) {
          const auto version = static_cast<Version>(index);
          CHECK(get_texture_for_version(version, x, y, z, 2) ==
                get_texture_for_version(version, x, y, z, 4) % 2);
        }
}

// Oracle-derived vectors for the other version modes (test/oracle/Dump.java
// query output, produced by the unmodified reference classes on a JVM).
TEST_CASE("get_texture_vanilla12 / Sodium / Sodium19 - oracle vectors") {
  CHECK(get_texture_vanilla12(0, 0, 0, 4) == 0);
  CHECK(get_texture_vanilla12(1, 0, 0, 4) == 3);
  CHECK(get_texture_vanilla12(-1, 0, 0, 4) == 0);
  CHECK(get_texture_vanilla12(0, 0, 1, 4) == 0);
  CHECK(get_texture_vanilla12(0, 0, -1, 4) == 0);
  CHECK(get_texture_vanilla12(-98765, 64, 43210, 4) == 1);
  CHECK(get_texture_vanilla12(-98765, 64, 43210, 2) == 1);

  CHECK(get_texture_sodium(0, 0, 0, 4) == 3);
  CHECK(get_texture_sodium(1, 0, 0, 4) == 0);
  CHECK(get_texture_sodium(-1, 0, 0, 4) == 3);
  CHECK(get_texture_sodium(0, 0, 1, 4) == 3);
  CHECK(get_texture_sodium(0, 0, -1, 4) == 1);
  CHECK(get_texture_sodium(-98765, 64, 43210, 4) == 1);
  CHECK(get_texture_sodium(-98765, 64, 43210, 2) == 1);

  CHECK(get_texture_sodium19(0, 0, 0, 4) == 2);
  CHECK(get_texture_sodium19(1, 0, 0, 4) == 1);
  CHECK(get_texture_sodium19(-1, 0, 0, 4) == 0);
  CHECK(get_texture_sodium19(0, 0, 1, 4) == 0);
  CHECK(get_texture_sodium19(0, 0, -1, 4) == 0);
  CHECK(get_texture_sodium19(-98765, 64, 43210, 4) == 3);
  CHECK(get_texture_sodium19(-98765, 64, 43210, 2) == 1);
}

TEST_CASE("get_texture_legacy - mod 4") {
  CHECK(get_texture_legacy(-1, 0, -1, 4) == 0);
  CHECK(get_texture_legacy(0, 0, -1, 4) == 1);
  CHECK(get_texture_legacy(1, 0, -1, 4) == 0);
  CHECK(get_texture_legacy(-2, 0, 0, 4) == 2);
  CHECK(get_texture_legacy(-1, 0, 0, 4) == 0);
  CHECK(get_texture_legacy(0, 0, 0, 4) == 0);
  CHECK(get_texture_legacy(1, 0, 0, 4) == 1);
  CHECK(get_texture_legacy(-1, 0, 1, 4) == 0);
  CHECK(get_texture_legacy(0, 0, 1, 4) == 3);
  CHECK(get_texture_legacy(1, 0, 1, 4) == 0);
}

TEST_CASE("get_texture_legacy - mod 2") {
  CHECK(get_texture_legacy(-6, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(-5, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(-4, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(-3, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(-2, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(-1, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(0, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(1, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(2, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(3, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(4, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(5, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(6, 0, 0, 2) == 0);
  CHECK(get_texture_legacy(7, 0, 0, 2) == 1);
  CHECK(get_texture_legacy(8, 0, 0, 2) == 0);
}

// Real-world verified formation: see test/fixtures/formation_a.expected.md.
TEST_CASE("formation_a matches at its verified origin") {
  const int OX = -108723, OY = -54, OZ = -69736;
  const int blocks[24][4] = {{1, 0, 0, 0}, {2, 0, 0, 2}, {0, 0, 1, 2}, {1, 0, 1, 3}, {2, 0, 1, 3},
                             {0, 0, 2, 3}, {1, 0, 2, 3}, {2, 0, 2, 0}, {3, 2, 2, 1}, {4, 2, 2, 2},
                             {0, 0, 3, 1}, {1, 0, 3, 2}, {2, 0, 3, 1}, {3, 2, 3, 2}, {4, 2, 3, 1},
                             {0, 0, 4, 1}, {1, 0, 4, 1}, {2, 0, 4, 0}, {3, 2, 4, 2}, {4, 2, 4, 0},
                             {0, 0, 5, 3}, {1, 0, 5, 3}, {2, 0, 5, 0}, {3, 2, 5, 3}};
  for (const auto& b : blocks)
    CHECK(get_texture_modern(OX + b[0], OY + b[1], OZ + b[2], 4) == b[3]);
}

TEST_CASE("rotate_block - coordinates cycle east -> south -> west -> north") {
  BlockInfo b{1, 5, 0, 0, false};
  rotate_block(b);
  CHECK(b.x == 0);
  CHECK(b.z == 1);
  rotate_block(b);
  CHECK(b.x == -1);
  CHECK(b.z == 0);
  rotate_block(b);
  CHECK(b.x == 0);
  CHECK(b.z == -1);
  rotate_block(b);
  CHECK(b.x == 1);
  CHECK(b.z == 0);
  CHECK(b.y == 5);
}

TEST_CASE("rotate_block - top rotations advance mod 4, side rotations flip parity") {
  BlockInfo top{2, 0, 3, 3, false};
  rotate_block(top);
  CHECK(top.rotation == 0);

  BlockInfo side{2, 0, 3, 1, true};
  rotate_block(side);
  CHECK(side.rotation == 0);
  rotate_block(side);
  CHECK(side.rotation == 1);

  // A side value must stay in {0, 1} after any number of rotations; the
  // old code let it reach 2-3, which a mod-2 texture can never equal.
  BlockInfo s{0, 0, 0, 1, true};
  for (int i = 0; i < 7; i++) {
    rotate_block(s);
    CHECK(s.rotation >= 0);
    CHECK(s.rotation <= 1);
  }
}

TEST_CASE("rotate_block - four rotations are the identity") {
  std::vector<BlockInfo> formation = {
      {1, 0, 0, 0, false}, {2, 1, -3, 3, false}, {-4, 2, 5, 1, true}, {0, -1, 7, 0, true}};
  std::vector<BlockInfo> original = formation;
  formation = rotated(formation, 4);
  for (size_t i = 0; i < formation.size(); i++) {
    CHECK(formation[i].x == original[i].x);
    CHECK(formation[i].y == original[i].y);
    CHECK(formation[i].z == original[i].z);
    CHECK(formation[i].rotation == original[i].rotation);
  }
}

namespace {
std::string write_temp_file(const std::string& content) {
  static int counter = 0;
  std::string path = "parser_test_" + std::to_string(counter++) + ".txt";
  std::ofstream out(path);
  out << content;
  return path;
}

// Writes, checks the parser throws, and cleans up the file.
void expect_parse_error(const std::string& content) {
  std::string path = write_temp_file(content);
  CHECK_THROWS_AS(parse_formation_file(path), std::runtime_error);
  std::remove(path.c_str());
}
} // namespace

TEST_CASE("parser - valid file with comments and blank lines") {
  std::string path = write_temp_file("# header comment\n1 2 3 2 0\n\n4 5 6 3 1\n");
  auto formation = parse_formation_file(path);
  REQUIRE(formation.size() == 2);
  CHECK(formation[0].x == 1);
  CHECK(formation[0].rotation == 2);
  CHECK(formation[0].is_side == false);
  CHECK(formation[1].is_side == true);
  // Side rotations are reduced to parity at parse time (reference: RotationInfo).
  CHECK(formation[1].rotation == 1);
  std::remove(path.c_str());
}

TEST_CASE("parser - rejects bad input loudly instead of skipping it") {
  CHECK_THROWS_AS(parse_formation_file("does_not_exist_12345.txt"), std::runtime_error);
  expect_parse_error("");
  expect_parse_error("# only a comment\n");
  expect_parse_error("1 2 3 4 0\n");  // rotation > 3
  expect_parse_error("1 2 3 -1 0\n"); // rotation < 0
  expect_parse_error("1 2 3 1 2\n");  // is_side not 0/1
  expect_parse_error("1 2 3\n");      // missing fields
  expect_parse_error("a b c d e\n");
  expect_parse_error("1 2 3 2 0 4 5 6 1 0\n");  // two blocks on one line
  expect_parse_error("1 2 3 2 0 oops\n");       // trailing garbage
  expect_parse_error("2147483647 0 0 0 0\n");   // absolute coordinate pasted as offset
  expect_parse_error("0 9999 0 0 0\n");         // y offset out of range
  expect_parse_error("1 2 3 0 0\n1 2 3 1 0\n"); // same block, contradictory tops
  expect_parse_error("1 2 3 2 0\n1 2 3 1 1\n"); // top=2 but side=1 (parity mismatch)
}

TEST_CASE("parser - consistent duplicates and top+side pairs are accepted") {
  // Exact duplicate collapses to one block.
  std::string path = write_temp_file("1 2 3 2 0\n1 2 3 2 0\n");
  CHECK(parse_formation_file(path).size() == 1);
  std::remove(path.c_str());

  // Same block observed as top (rot 2) and side (parity 0): consistent.
  path = write_temp_file("1 2 3 2 0\n1 2 3 0 1\n# trailing comment ok\n4 5 6 1 0 # inline comment\n");
  auto f = parse_formation_file(path);
  CHECK(f.size() == 3);
  std::remove(path.c_str());
}

// The bitplane planner and the fill, filter and survivor check are shared
// host code, so what every backend runs is checked here without a GPU.
namespace {
using namespace bitplanes;

const SearchBounds PLAN_BOUNDS{-70, 100, -3, 2, -37, 41};

// Two top faces on every z coset, with offsets on both sides of the origin.
std::vector<BlockInfo> plan_formation() {
  std::vector<BlockInfo> blocks;
  for (int i = 0; i < 2 * STRIDE; ++i)
    blocks.push_back({13 * i - 60, i % 3 - 1, i - 4, i % 4, false});
  blocks.push_back({7, 2, 3, 1, true});
  return blocks;
}
} // namespace

TEST_CASE("bitplanes - residue is the non-negative remainder") {
  for (int z = -12; z <= 12; ++z) {
    CHECK(residue(z) >= 0);
    CHECK(residue(z) < STRIDE);
    CHECK((z - residue(z)) % STRIDE == 0);
  }
}

TEST_CASE("bitplanes - compact levels list each needed world y once, ascending") {
  const SearchBounds bounds{0, 0, 10, 12, 0, 0};
  CHECK(compact_levels(bounds, {-1, 0, 5}) == std::vector<int>{9, 10, 11, 12, 15, 16, 17});
}

TEST_CASE("bitplanes - eligibility needs four filter bits on every z coset") {
  auto blocks = plan_formation();
  CHECK(eligible(blocks));
  blocks.erase(blocks.begin()); // leaves one coset with a single top face
  CHECK_FALSE(eligible(blocks));
}

TEST_CASE("bitplanes - a budget too small for one slab gives no plan") {
  const std::vector<std::vector<BlockInfo>> formations{plan_formation()};
  CHECK_FALSE(plan_search(PLAN_BOUNDS, formations, 1).has_value());
  CHECK(plan_search(PLAN_BOUNDS, formations, std::size_t{1} << 30).has_value());
}

TEST_CASE("bitplanes - a slab has no more rows than the caller allows") {
  const std::vector<std::vector<BlockInfo>> formations{plan_formation()};
  const auto plan = plan_search(PLAN_BOUNDS, formations, std::size_t{1} << 30, 3);
  REQUIRE(plan.has_value());
  CHECK(plan->slab_rows == 3);
  CHECK(plan->slab(0).layout.nz == 3);
  CHECK_FALSE(plan_search(PLAN_BOUNDS, formations, std::size_t{1} << 30, 0).has_value());
}

TEST_CASE("bitplanes - every block is a plane check on its own coset and verified on the others") {
  const auto blocks = plan_formation();
  const std::vector<std::vector<BlockInfo>> formations{blocks};
  const auto plan = plan_search(PLAN_BOUNDS, formations, std::size_t{1} << 30);
  REQUIRE(plan.has_value());
  const auto& direction = plan->directions.front();
  CHECK(direction.checks.size() == blocks.size());
  CHECK(direction.verify.size() == blocks.size() * (STRIDE - 1));
  for (int coset = 0; coset < STRIDE; ++coset) {
    const int on_plane = direction.offsets[coset + 1] - direction.offsets[coset];
    const int verified = direction.verify_offsets[coset + 1] - direction.verify_offsets[coset];
    CHECK(on_plane + verified == static_cast<int>(blocks.size()));
  }
}

// Replays the index arithmetic of fill_planes and match_words: the field bit a
// plane check reads for a candidate must be the one that was hashed for the
// candidate's position plus the block offset.
TEST_CASE("bitplanes - plane checks read the block's own position in every slab") {
  const auto blocks = plan_formation();
  const std::vector<std::vector<BlockInfo>> formations{blocks};
  const auto roomy = plan_search(PLAN_BOUNDS, formations, std::size_t{1} << 30);
  REQUIRE(roomy.has_value());
  // Room for seven hashed rows, so the 79 z rows need several slabs.
  const auto row_bytes = roomy->plane_y.size() * roomy->field_words * sizeof(Planes);
  const auto plan = plan_search(PLAN_BOUNDS, formations, 7 * row_bytes);
  REQUIRE(plan.has_value());
  CHECK(plan->slab_rows < static_cast<int>(PLAN_BOUNDS.nz()));

  const auto& direction = plan->directions.front();
  std::vector<PlaneCheck> slab_checks(direction.checks.size());
  std::int64_t rows_covered = 0;
  for (std::int64_t z_offset = 0; z_offset < static_cast<std::int64_t>(PLAN_BOUNDS.nz());) {
    const auto slab = plan->slab(z_offset);
    const auto& layout = slab.layout;
    REQUIRE(layout.nz > 0);
    CHECK(layout.bounds.z_min == PLAN_BOUNDS.z_min + z_offset);
    CHECK(slab.field_words <= plan->plane_words);
    prepare_plane_checks(layout, direction.checks, direction.offsets.data(), slab_checks);

    for (int row = 0; row < layout.nz; ++row)
      for (int level = 0; level < layout.ny; ++level)
        for (int word = 0; word < layout.candidate_words; ++word) {
          const int z = layout.bounds.z_min + row;
          const int coset = residue(z);
          const std::size_t base =
              (static_cast<std::size_t>(level) * layout.hashed_rows + row / STRIDE) * layout.field_words +
              word;
          for (int i = direction.offsets[coset]; i < direction.offsets[coset + 1]; ++i) {
            const auto& check = direction.checks[i];
            const std::size_t index = base + slab_checks[i].offset;
            REQUIRE(index + 1 < slab.field_words);
            const auto field_word = index % layout.field_words;
            const auto field_row = index / layout.field_words;
            const auto hashed_z =
                layout.first_hashed_z + static_cast<std::int64_t>(field_row % layout.hashed_rows) * STRIDE;
            const int hashed_y = plan->plane_y[field_row / layout.hashed_rows];
            const auto hashed_x =
                layout.field_x + static_cast<std::int64_t>(field_word) * WORD_BITS + slab_checks[i].shift;
            const int block_x = check.x_offset + plan->min_x;
            const int block_y = plan->plane_y[check.y_slot] - PLAN_BOUNDS.y_min;
            REQUIRE(hashed_z == z + check.z_offset);
            REQUIRE(hashed_y == PLAN_BOUNDS.y_min + level + block_y);
            REQUIRE(hashed_x == PLAN_BOUNDS.x_min + word * WORD_BITS + block_x);
          }
        }
    rows_covered += layout.nz;
    z_offset += layout.nz;
  }
  CHECK(rows_covered == static_cast<std::int64_t>(PLAN_BOUNDS.nz()));
}

TEST_CASE("bitplanes - a side face shows the low bit of the top-face rotation in every version") {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> coordinate(-30000000, 30000000);
  for (int version = 0; version < NUM_VERSIONS; ++version)
    for (int i = 0; i < 20000; ++i) {
      const int x = coordinate(rng), y = coordinate(rng) % 400, z = coordinate(rng);
      const auto v = static_cast<Version>(version);
      REQUIRE(get_texture_for_version(v, x, y, z, MOD_SIDE) ==
              (get_texture_for_version(v, x, y, z, MOD_TOP_BOTTOM) & 1));
    }
}

TEST_CASE("bitplanes - field, filter and survivor check find exactly the brute-force matches") {
  // Three words of x with a partial last word, slabs of 5 rows so every coset
  // meets a slab edge, and few enough blocks that most masks keep survivors.
  const SearchBounds bounds{-70, 100, -3, 2, -17, 21};
  const MatchResult origin{-5, 0, 3};
  for (int version = 0; version < NUM_VERSIONS; ++version) {
    const auto v = static_cast<Version>(version);
    std::vector<BlockInfo> blocks{{0, 0, 0, 0, false}, {70, 1, 2, 0, false}, {-65, -1, -3, 0, true},
                                  {3, 2, 5, 0, true},  {1, 0, 6, 0, false},  {-9, 1, 4, 0, true}};
    for (auto& block : blocks)
      block.rotation = get_texture_for_version(v, origin.x + block.x, origin.y + block.y, origin.z + block.z,
                                               block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM);
    const auto matches_at = [&](int x, int y, int z, bool lattice_only) {
      return std::all_of(blocks.begin(), blocks.end(), [&](const BlockInfo& block) {
        if (lattice_only && residue(std::int64_t{z} + block.z) != 0)
          return true;
        return get_texture_for_version(v, x + block.x, y + block.y, z + block.z,
                                       block.is_side ? MOD_SIDE : MOD_TOP_BOTTOM) == block.rotation;
      });
    };

    const std::vector<std::vector<BlockInfo>> formations{blocks};
    auto plan = plan_search(bounds, formations, std::size_t{1} << 30);
    REQUIRE(plan.has_value());
    plan->slab_rows = 5;
    const auto& direction = plan->directions.front();
    std::vector<Planes> field;
    std::vector<PlaneCheck> checks(direction.checks.size());
    std::uint64_t found = 0;
    bool found_origin = false;

    for (std::int64_t z_offset = 0; z_offset < static_cast<std::int64_t>(bounds.nz());) {
      const auto layout = plan->slab(z_offset).layout;
      field.clear();
      for (const int y : plan->plane_y)
        for (int z_index = 0; z_index < layout.hashed_rows; ++z_index)
          for (int word = 0; word < layout.field_words; ++word)
            field.push_back(field_word(v, layout, y, z_index, word));
      prepare_plane_checks(layout, direction.checks, direction.offsets.data(), checks);
      std::vector<Word> row_masks(static_cast<std::size_t>(layout.candidate_words));

      for (int y_row = 0; y_row < layout.ny; ++y_row)
        for (int z_row = 0; z_row < layout.nz; ++z_row)
          for (int word = 0; word < layout.candidate_words; ++word) {
            if (word == 0)
              cpu_bitplanes::filter_row(layout, checks.data(), direction.offsets.data(), field.data(),
                                        static_cast<std::size_t>(y_row), static_cast<std::size_t>(z_row),
                                        row_masks.data());
            const int y = layout.bounds.y_min + y_row, z = layout.bounds.z_min + z_row;
            const int coset = residue(z);
            const Word mask =
                filter_word(layout, checks.data(), direction.offsets.data(), field.data(),
                            static_cast<std::size_t>(y_row), static_cast<std::size_t>(z_row), word);
            REQUIRE(row_masks[word] == mask); // the CPU backend's row filter
            for (int bit = 0; bit < WORD_BITS; ++bit) {
              const int x = layout.bounds.x_min + word * WORD_BITS + bit;
              const bool survives = (mask >> bit) & 1;
              if (x > bounds.x_max) {
                REQUIRE(!survives);
                continue;
              }
              REQUIRE(survives == matches_at(x, y, z, true));
              const bool matches =
                  survives && survivor_matches(v, direction.verify.data(), direction.verify_offsets[coset],
                                               direction.verify_offsets[coset + 1],
                                               static_cast<std::uint32_t>(x) * 3129871u,
                                               std::int64_t{z} * 116129781LL, y);
              REQUIRE(matches == matches_at(x, y, z, false));
              found += matches;
              found_origin |= matches && x == origin.x && y == origin.y && z == origin.z;
            }
          }
      z_offset += layout.nz;
    }
    CHECK(found_origin);
    CHECK(found >= 1);
  }
}
