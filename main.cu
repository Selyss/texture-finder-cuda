#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

constexpr int MOD_TOP_BOTTOM = 4;
constexpr int MOD_SIDE = 2;

constexpr int MODERN_VERSION = 0;

struct BlockInfo
{
    int x, y, z;
    int rotation;
    bool isSide;
};

// TODO: implement the other versions of the texture rotation generation

// 1.13 - 1.21.1
__device__ int getTextureLegacy(int x, int y, int z, int mod)
{
    constexpr int64_t multiplier = 0x5DEECE66DLL;
    constexpr int64_t mask = (1LL << 48) - 1;
    int64_t l = (static_cast<int64_t>(x * 3129871)) ^
                (static_cast<int64_t>(z) * 116129781L) ^ static_cast<int64_t>(y);
    l = l * l * 42317861L + l * 11L;
    int64_t seed = l >> 16;
    seed = (seed ^ multiplier) & mask;
    int rand = static_cast<int>((seed * 0xBB20B4600A69L + 0x40942DE6BAL) >> 16);
    return abs(rand) % mod;
}

// 1.21.2+
__device__ int getTextureModern(int x, int y, int z, int mod)
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

__host__ __device__ void rotateFormation(BlockInfo &block)
{
    int temp = block.x;
    block.x = -block.z;
    block.z = temp;

    if (block.rotation == 3)
    {
        block.rotation = 0;
    }
    else
    {
        block.rotation++;
    }
}

__global__ void matchFormationKernel(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, BlockInfo *tops_and_bottoms, int tb_size, BlockInfo *sides, int side_size, int version)
{
    int x = x_min + blockIdx.x * blockDim.x + threadIdx.x;
    int y = y_min + blockIdx.y * blockDim.y + threadIdx.y;
    int z = z_min + blockIdx.z * blockDim.z + threadIdx.z;

    // FIXME: entering the same y coord to scan a single level does not work
    if (x > x_max || y > y_max || z > z_max)
        return;

    bool match = true;

    for (int i = 0; i < tb_size; i++)
    {
        int bx = x + tops_and_bottoms[i].x;
        int by = y + tops_and_bottoms[i].y;
        int bz = z + tops_and_bottoms[i].z;
        int texture = (version == MODERN_VERSION) ? getTextureModern(bx, by, bz, MOD_TOP_BOTTOM) : getTextureLegacy(bx, by, bz, MOD_TOP_BOTTOM);

        // this is done instead of an if statement for performance reasons
        match &= (tops_and_bottoms[i].rotation == texture);
        if (!match)
            break;
    }

    // NOTE: is this needed along with the other if?
    if (!match)
        return;

    for (int i = 0; i < side_size; i++)
    {
        int cx = x + sides[i].x;
        int cy = y + sides[i].y;
        int cz = z + sides[i].z;

        int texture = (version == MODERN_VERSION) ? getTextureModern(cx, cy, cz, MOD_SIDE) : getTextureLegacy(cx, cy, cz, MOD_SIDE);

        match &= (sides[i].rotation == texture);

        if (!match)
            break;
    }

    if (match)
    {
        printf("Match found at [%d, %d, %d]\n", x, y, z);
    }
}

std::vector<BlockInfo> parseFormationFile(const std::string &filename)
{
    std::vector<BlockInfo> formation;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        BlockInfo info;

        if (iss >> info.x >> info.y >> info.z >> info.rotation >> info.isSide)
        {
            formation.push_back(info);
        }
    }
    std::cout << formation.size() << " blocks read from formation file: " << filename << std::endl;
    return formation;
}

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

    BlockInfo *d_topsAndBottoms, *d_sides;

    cudaMallocManaged(&d_topsAndBottoms, topsAndBottoms.size() * sizeof(BlockInfo));
    cudaMallocManaged(&d_sides, sides.size() * sizeof(BlockInfo));

    // FIXME: is this needed or do I cudaMemcpy?
    std::memcpy(d_topsAndBottoms, topsAndBottoms.data(), topsAndBottoms.size() * sizeof(BlockInfo));
    std::memcpy(d_sides, sides.data(), sides.size() * sizeof(BlockInfo));

    // FIXME: this needs to be optimized for the specific hardware the program is running on
    dim3 threadsPerBlock(8, 2, 8);

    dim3 numBlocks(
        ((x_max - x_min) + threadsPerBlock.x - 1) / threadsPerBlock.x,
        ((y_max - y_min) + threadsPerBlock.y - 1) / threadsPerBlock.y,
        ((z_max - z_min) + threadsPerBlock.z - 1) / threadsPerBlock.z);

    matchFormationKernel<<<numBlocks, threadsPerBlock>>>(x_min, x_max, y_min, y_max, z_min, z_max, d_topsAndBottoms, topsAndBottoms.size(), d_sides, sides.size(), version);
    cudaDeviceSynchronize();

    cudaFree(d_topsAndBottoms);
    cudaFree(d_sides);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Search completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
