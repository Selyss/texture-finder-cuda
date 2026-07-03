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
//   {legacy,modern}_{top,side}.bin  one raw byte per coordinate,
//       offset = (yIndex*1024 + z+512)*1024 + x+512,
//       yIndex over {-64,-54,0,63,255,319}, z,x in [-512,511]
//   extremes.txt / random.txt  lines: "x y z legacy_top legacy_side modern_top modern_side"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "texture.cuh"

namespace
{
long long g_mismatches = 0;
long long g_compared = 0;

void report(const char *what, int x, int y, int z, int expected, int got)
{
    if (g_mismatches < 10)
        std::fprintf(stderr, "MISMATCH %s at (%d, %d, %d): oracle=%d ours=%d\n",
                     what, x, y, z, expected, got);
    g_mismatches++;
}

void checkValue(const char *what, int x, int y, int z, int oracle, int ours)
{
    g_compared++;
    if (oracle != ours)
        report(what, x, y, z, oracle, ours);
}

bool diffGrid(const std::string &path, const char *what, bool modern, int mod)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const int Y_VALUES[6] = {-64, -54, 0, 63, 255, 319};
    if (data.size() != 1024ull * 1024ull * 6ull)
    {
        std::fprintf(stderr, "%s: unexpected size %zu\n", path.c_str(), data.size());
        return false;
    }
    size_t i = 0;
    for (int yi = 0; yi < 6; yi++)
        for (int z = -512; z <= 511; z++)
            for (int x = -512; x <= 511; x++, i++)
            {
                const int oracle = (unsigned char)data[i];
                const int ours = modern ? getTextureModern(x, Y_VALUES[yi], z, mod)
                                        : getTextureLegacy(x, Y_VALUES[yi], z, mod);
                checkValue(what, x, Y_VALUES[yi], z, oracle, ours);
            }
    return true;
}

bool diffText(const std::string &path, long long expectedLines)
{
    std::ifstream f(path);
    if (!f)
    {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }
    int x, y, z, lt, ls, mt, ms;
    long long lines = 0;
    while (f >> x >> y >> z >> lt >> ls >> mt >> ms)
    {
        checkValue("legacy_top", x, y, z, lt, getTextureLegacy(x, y, z, 4));
        checkValue("legacy_side", x, y, z, ls, getTextureLegacy(x, y, z, 2));
        checkValue("modern_top", x, y, z, mt, getTextureModern(x, y, z, 4));
        checkValue("modern_side", x, y, z, ms, getTextureModern(x, y, z, 2));
        lines++;
    }
    std::printf("%s: %lld lines\n", path.c_str(), lines);
    // A malformed line stops the >> loop early; partial coverage must not
    // pass as success.
    if (!f.eof())
    {
        std::fprintf(stderr, "%s: stopped at a malformed line (after %lld lines)\n",
                      path.c_str(), lines);
        return false;
    }
    if (lines != expectedLines)
    {
        std::fprintf(stderr, "%s: expected %lld lines, got %lld\n", path.c_str(),
                      expectedLines, lines);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <dumps_dir>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];

    bool ok = true;
    ok &= diffGrid(dir + "/legacy_top.bin", "legacy_top", false, 4);
    ok &= diffGrid(dir + "/legacy_side.bin", "legacy_side", false, 2);
    ok &= diffGrid(dir + "/modern_top.bin", "modern_top", true, 4);
    ok &= diffGrid(dir + "/modern_side.bin", "modern_side", true, 2);
    ok &= diffText(dir + "/extremes.txt", 3072);
    ok &= diffText(dir + "/random.txt", 1000000);

    std::printf("Compared %lld values, %lld mismatches\n", g_compared, g_mismatches);
    if (!ok || g_mismatches > 0)
        return 1;
    std::printf("OK: C++ implementations match the Java oracle exactly\n");
    return 0;
}
