// Generates a synthetic formation file whose blocks are guaranteed to match
// at a chosen origin, for end-to-end testing of the searcher. The emitted
// file is expressed in the "recreation frame" for the given direction: after
// the searcher rotates it <direction> times, it equals the ground truth.
//
// Usage: gen_formation <version 0-4> <direction 0-3> <ox> <oy> <oz> <count> <sidePercent> <seed>
// Output on stdout; '#' header lines record the parameters.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "texture/formation.hpp"
#include "texture/rotation.hpp"
#include "texture/texture_rules.hpp"

int main(int argc, char* argv[]) {
  if (argc != 9) {
    std::fprintf(stderr,
                 "Usage: %s <version 0-4> <direction 0-3> <ox> <oy> <oz> <count> <sidePercent> <seed>\n",
                 argv[0]);
    return 1;
  }
  const int version = std::atoi(argv[1]);
  const int direction = std::atoi(argv[2]);
  const int ox = std::atoi(argv[3]);
  const int oy = std::atoi(argv[4]);
  const int oz = std::atoi(argv[5]);
  const int count = std::atoi(argv[6]);
  const int side_percent = std::atoi(argv[7]);
  const unsigned seed = (unsigned)std::strtoul(argv[8], nullptr, 10);

  if (version < 0 || version >= NUM_VERSIONS || direction < 0 || direction > 3 || count < 1 ||
      count > MAX_FORMATION_BLOCKS || side_percent < 0 || side_percent > 100) {
    std::fprintf(stderr, "Invalid arguments\n");
    return 1;
  }

  // std::mt19937 gives the same numbers everywhere, but the standard
  // distributions do not, so the range is taken by hand. A seed then makes
  // the same formation on every platform.
  std::mt19937 rng(seed);
  const auto draw = [&rng](int low, int high) {
    return low + static_cast<int>(rng() % static_cast<unsigned>(high - low + 1));
  };

  // Ground truth in the world frame: real texture values at origin+offset.
  std::set<std::tuple<int, int, int>> used;
  std::vector<BlockInfo> truth;
  while ((int)truth.size() < count) {
    const int bx = draw(-8, 8);
    const int by = draw(-2, 3);
    const int bz = draw(-8, 8);
    if (!used.insert({bx, by, bz}).second)
      continue;
    BlockInfo b;
    b.x = bx;
    b.y = by;
    b.z = bz;
    b.is_side = draw(0, 99) < side_percent;
    const int mod = b.is_side ? MOD_SIDE : MOD_TOP_BOTTOM;
    b.rotation = get_texture_for_version(static_cast<Version>(version), ox + bx, oy + by, oz + bz, mod);
    truth.push_back(b);
  }

  // The searcher turns the file it reads <direction> times; emitting truth
  // turned (4 - direction) times round-trips to the world frame exactly.
  truth = rotated(std::move(truth), (4 - direction) % 4);

  std::printf(
      "# synthetic formation: version=%d direction=%d origin=%d,%d,%d count=%d sidePercent=%d seed=%u\n",
      version, direction, ox, oy, oz, count, side_percent, seed);
  for (const auto& b : truth)
    std::printf("%d %d %d %d %d\n", b.x, b.y, b.z, b.rotation, b.is_side ? 1 : 0);
  return 0;
}
