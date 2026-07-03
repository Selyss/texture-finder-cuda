#include "parser.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
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

    std::vector<BlockInfo> formation;
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

        BlockInfo info;
        int isSide;
        if (!(iss >> info.x >> info.y >> info.z >> info.rotation >> isSide))
            throw std::runtime_error(filename + ":" + std::to_string(lineNo) +
                                     ": expected \"x y z rotation isSide\", got: " + line);
        if (isSide != 0 && isSide != 1)
            throw std::runtime_error(filename + ":" + std::to_string(lineNo) +
                                     ": isSide must be 0 or 1");
        if (info.rotation < 0 || info.rotation > 3)
            throw std::runtime_error(filename + ":" + std::to_string(lineNo) +
                                     ": rotation must be 0-3");
        info.isSide = isSide != 0;
        if (info.isSide)
            info.rotation %= 2;
        formation.push_back(info);
    }

    if (formation.empty())
        throw std::runtime_error("No blocks in formation file: " + filename);

    std::cout << formation.size() << " blocks read from formation file: " << filename << std::endl;
    return formation;
}
