#include "parser.cuh"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

// Format: one block per line, "x y z rotation isSide", blank lines and lines
// starting with '#' ignored. rotation is the 0-3 value from the Textures to
// Numbers pack; for side faces only its parity is meaningful (the game rolls
// one of 4 variants and a side face exposes rotation % 2), matching the
// reference RotationInfo.
std::vector<BlockInfo> parseFormationFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Cannot open formation file: " + filename);

    // Offsets are relative to a nearby origin block; huge values are a typo
    // (an absolute coordinate pasted in) and would overflow kernel math.
    constexpr int OFFSET_LIMIT_XZ = 1'000'000;
    constexpr int OFFSET_LIMIT_Y = 4'096;

    struct SeenFaces
    {
        int topRot = -1;  // -1 = not seen yet
        int sideRot = -1;
    };
    std::vector<BlockInfo> formation;
    std::map<std::tuple<int, int, int>, SeenFaces> seen;
    std::string line;
    int lineNo = 0;
    while (std::getline(file, line))
    {
        lineNo++;
        std::istringstream iss(line);

        std::string first;
        if (!(iss >> first) || first[0] == '#')
            continue;
        iss.clear();
        iss.str(line);

        const auto fail = [&](const std::string &why) {
            throw std::runtime_error(filename + ":" + std::to_string(lineNo) + ": " + why);
        };

        BlockInfo info;
        int isSide;
        if (!(iss >> info.x >> info.y >> info.z >> info.rotation >> isSide))
            fail("expected \"x y z rotation isSide\", got: " + line);
        std::string extra;
        if (iss >> extra && extra[0] != '#')
            fail("unexpected trailing token \"" + extra + "\" (one block per line)");
        if (isSide != 0 && isSide != 1)
            fail("isSide must be 0 or 1");
        if (info.rotation < 0 || info.rotation > 3)
            fail("rotation must be 0-3");
        if (info.x < -OFFSET_LIMIT_XZ || info.x > OFFSET_LIMIT_XZ ||
            info.z < -OFFSET_LIMIT_XZ || info.z > OFFSET_LIMIT_XZ ||
            info.y < -OFFSET_LIMIT_Y || info.y > OFFSET_LIMIT_Y)
            fail("block offset out of range (offsets are relative to the origin block)");
        info.isSide = isSide != 0;
        if (info.isSide)
            info.rotation %= 2;

        // A block may be listed once as a top face and once as a side face
        // (consistent iff side == top % 2, since the side exposes the parity
        // of the same roll), but contradictory entries can never match any
        // origin and would silently produce zero results.
        SeenFaces &faces = seen[{info.x, info.y, info.z}];
        int &sameFace = info.isSide ? faces.sideRot : faces.topRot;
        const int otherFace = info.isSide ? faces.topRot : faces.sideRot;
        if (sameFace >= 0)
        {
            if (sameFace != info.rotation)
                fail("conflicts with an earlier entry for the same block; no origin could match both");
            continue; // identical duplicate: harmless, keep one copy
        }
        if (otherFace >= 0)
        {
            const int top = info.isSide ? otherFace : info.rotation;
            const int side = info.isSide ? info.rotation : otherFace;
            if (top % 2 != side)
                fail("top and side entries for the same block disagree (side must equal top % 2)");
        }
        sameFace = info.rotation;
        formation.push_back(info);
    }

    if (formation.empty())
        throw std::runtime_error("No blocks in formation file: " + filename);

    std::cout << formation.size() << " blocks read from formation file: " << filename << std::endl;
    return formation;
}
