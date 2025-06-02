#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "blockinfo.cuh"
#include "texture.cuh"
#include "rotation.cuh"
#include "kernel.cuh"
#include "parser.cuh"

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    if (argc != 10)
    {
        std::cerr << "Usage: " << argv[0] << " <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction>";
        return 1;
    }

    // TODO: validate input
    int x_min = std::stoi(argv[1]);
    int x_max = std::stoi(argv[2]);

    int y_min = std::stoi(argv[3]);
    int y_max = std::stoi(argv[4]);

    int z_min = std::stoi(argv[5]);
    int z_max = std::stoi(argv[6]);

    // getTextureLegacy is 1
    // getTextureModern is 0
    int version = std::stoi(argv[7]);

    std::string file = argv[8];
    int direction = std::stoi(argv[9]);

    if (version == MODERN_VERSION)
    {
        std::cout << "Version: 1.21.2+" << std::endl;
    }
    else
    {
        std::cout << "Version: 1.13 - 1.21.1" << std::endl;
    }

    std::vector<BlockInfo> formation = parseFormationFile(file);

    std::vector<BlockInfo> topsAndBottoms, sides;

    for (const auto &info : formation)
    {
        if (info.isSide)
        {
            sides.push_back(info);
        }
        else
        {
            topsAndBottoms.push_back(info);
        }
    }

    switch (direction)
    {
    case 0:
        std::cout << "Facing: North" << std::endl;
        break;
    case 1:
        std::cout << "Facing: West" << std::endl;
        break;
    case 2:
        std::cout << "Facing: South" << std::endl;
        break;
    case 3:
        std::cout << "Facing: East" << std::endl;
        break;
    default:
        std::cerr << "Invalid direction: " << direction << std::endl;
        return 1;
    }

    for (int x = 0; x < direction; x++)
    {
        for (int i = 0; i < topsAndBottoms.size(); i++)
        {
            rotateFormation(topsAndBottoms[i]);
        }
        for (int i = 0; i < sides.size(); i++)
        {
            rotateFormation(sides[i]);
        }
    }

    cudaMemcpyToSymbol(d_topsAndBottoms, topsAndBottoms.data(), topsAndBottoms.size() * sizeof(BlockInfo));
    cudaMemcpyToSymbol(d_sides, sides.data(), sides.size() * sizeof(BlockInfo));

    // FIXME: this needs to be optimized for the specific hardware the program is running on
    dim3 threadsPerBlock(8, 2, 8);

    dim3 numBlocks(
        ((x_max - x_min) + threadsPerBlock.x - 1) / threadsPerBlock.x,
        ((y_max - y_min) + threadsPerBlock.y) / threadsPerBlock.y,
        ((z_max - z_min) + threadsPerBlock.z - 1) / threadsPerBlock.z);

    matchFormationKernel<<<numBlocks, threadsPerBlock>>>(x_min, x_max, y_min, y_max, z_min, z_max, topsAndBottoms.size(), sides.size(), version);
    cudaDeviceSynchronize();

    cudaFree(d_topsAndBottoms);
    cudaFree(d_sides);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Search completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
