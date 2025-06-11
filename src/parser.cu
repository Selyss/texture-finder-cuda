#include "parser.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<BlockInfo> parseFormationFile(const std::string &filename)
{
    std::vector<BlockInfo> formation;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        BlockInfo info;

        if (!(iss >> info.x >> info.y >> info.z >> info.rotation >> info.isSide))
        {
            std::cerr << "Invalid format in formation file: " << filename
                      << std::endl;
            continue;
        }
        if (info.isSide)
        {
            info.rotation %= 2;
        }
        formation.push_back(info);
    }
    std::cout << formation.size() << " blocks read from formation file: " << filename << std::endl;
    return formation;
}