// Differential test against the Java oracle dumps (generated on the GPU box
// by /root/oracle/regen.sh from the reference implementation). Compares every
// dumped value with this repo's C++ implementations. Any mismatch means the
// C++ diverged from Java semantics — including future Minecraft RNG changes
// once the oracle is regenerated against a newer reference.
//
// Usage: oracle_diff <dumps_dir>
// Exits 0 and prints totals when everything matches; exits 1 with the first
// mismatches otherwise.
//
// Dump formats (see dumps/README.md):
//   {legacy,modern,vanilla12,sodium,sodium19}_{top,side}.bin  one raw byte per coordinate,
//       offset = (y_index*1024 + z+512)*1024 + x+512,
//       y_index over {-64,-54,0,63,255,319}, z,x in [-512,511]
//   extremes.txt / random.txt  lines: "x y z" + top/side pairs for
//       legacy, modern, vanilla12, sodium, sodium19 (10 value columns)

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "texture/texture_rules.hpp"

namespace {
long long g_mismatches = 0;
long long g_compared = 0;

void report(const char* what, int x, int y, int z, int expected, int got) {
  if (g_mismatches < 10)
    std::fprintf(stderr, "MISMATCH %s at (%d, %d, %d): oracle=%d ours=%d\n", what, x, y, z, expected, got);
  g_mismatches++;
}

void check_value(const char* what, int x, int y, int z, int oracle, int ours) {
  g_compared++;
  if (oracle != ours)
    report(what, x, y, z, oracle, ours);
}

bool diff_grid(const std::string& path, const char* what, int version, int mod) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "Cannot open %s\n", path.c_str());
    return false;
  }
  std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const int Y_VALUES[6] = {-64, -54, 0, 63, 255, 319};
  if (data.size() != 1024ull * 1024ull * 6ull) {
    std::fprintf(stderr, "%s: unexpected size %zu\n", path.c_str(), data.size());
    return false;
  }
  size_t i = 0;
  for (int yi = 0; yi < 6; yi++)
    for (int z = -512; z <= 511; z++)
      for (int x = -512; x <= 511; x++, i++) {
        const int oracle = (unsigned char)data[i];
        const int ours = get_texture_for_version(version, x, Y_VALUES[yi], z, mod);
        check_value(what, x, Y_VALUES[yi], z, oracle, ours);
      }
  return true;
}

bool diff_text(const std::string& path, long long expected_lines) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "Cannot open %s\n", path.c_str());
    return false;
  }
  int x, y, z, lt, ls, mt, ms, vt, vs, st, ss, nt, ns;
  long long lines = 0;
  while (f >> x >> y >> z >> lt >> ls >> mt >> ms >> vt >> vs >> st >> ss >> nt >> ns) {
    check_value("legacy_top", x, y, z, lt, get_texture_legacy(x, y, z, 4));
    check_value("legacy_side", x, y, z, ls, get_texture_legacy(x, y, z, 2));
    check_value("modern_top", x, y, z, mt, get_texture_modern(x, y, z, 4));
    check_value("modern_side", x, y, z, ms, get_texture_modern(x, y, z, 2));
    check_value("vanilla12_top", x, y, z, vt, get_texture_vanilla12(x, y, z, 4));
    check_value("vanilla12_side", x, y, z, vs, get_texture_vanilla12(x, y, z, 2));
    check_value("sodium_top", x, y, z, st, get_texture_sodium(x, y, z, 4));
    check_value("sodium_side", x, y, z, ss, get_texture_sodium(x, y, z, 2));
    check_value("sodium19_top", x, y, z, nt, get_texture_sodium19(x, y, z, 4));
    check_value("sodium19_side", x, y, z, ns, get_texture_sodium19(x, y, z, 2));
    lines++;
  }
  std::printf("%s: %lld lines\n", path.c_str(), lines);
  // A malformed line stops the >> loop early; partial coverage must not
  // pass as success.
  if (!f.eof()) {
    std::fprintf(stderr, "%s: stopped at a malformed line (after %lld lines)\n", path.c_str(), lines);
    return false;
  }
  if (lines != expected_lines) {
    std::fprintf(stderr, "%s: expected %lld lines, got %lld\n", path.c_str(), expected_lines, lines);
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s <dumps_dir>\n", argv[0]);
    return 1;
  }
  const std::string dir = argv[1];

  bool ok = true;
  ok &= diff_grid(dir + "/legacy_top.bin", "legacy_top", 1, 4);
  ok &= diff_grid(dir + "/legacy_side.bin", "legacy_side", 1, 2);
  ok &= diff_grid(dir + "/modern_top.bin", "modern_top", 0, 4);
  ok &= diff_grid(dir + "/modern_side.bin", "modern_side", 0, 2);
  ok &= diff_grid(dir + "/vanilla12_top.bin", "vanilla12_top", 2, 4);
  ok &= diff_grid(dir + "/vanilla12_side.bin", "vanilla12_side", 2, 2);
  ok &= diff_grid(dir + "/sodium_top.bin", "sodium_top", 3, 4);
  ok &= diff_grid(dir + "/sodium_side.bin", "sodium_side", 3, 2);
  ok &= diff_grid(dir + "/sodium19_top.bin", "sodium19_top", 4, 4);
  ok &= diff_grid(dir + "/sodium19_side.bin", "sodium19_side", 4, 2);
  ok &= diff_text(dir + "/extremes.txt", 3072);
  ok &= diff_text(dir + "/random.txt", 1000000);

  std::printf("Compared %lld values, %lld mismatches\n", g_compared, g_mismatches);
  if (!ok || g_mismatches > 0)
    return 1;
  std::printf("OK: C++ implementations match the Java oracle exactly\n");
  return 0;
}
