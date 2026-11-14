#include "checkpoint.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "session.hpp"

namespace
{
std::uint64_t fnv1a(const void *data, std::size_t n, std::uint64_t h = 0xCBF29CE484222325ull)
{
    const unsigned char *p = (const unsigned char *)data;
    for (std::size_t i = 0; i < n; i++)
    {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}
} // namespace

std::uint64_t checkpointSignature(const SearchRequest &request)
{
    const auto &bounds = request.bounds;
    const auto &formation = request.formation;
    const auto &directions = request.directions;
    const auto version = request.version;
    std::string dirsKey;
    for (int d : directions)
        dirsKey += std::to_string(d) + ",";
    std::ostringstream sig;
    sig << bounds.x_min << " " << bounds.x_max << " " << bounds.y_min << " " << bounds.y_max
        << " " << bounds.z_min << " " << bounds.z_max << " " << version << " " << dirsKey << "\n";
    for (const auto &b : formation)
        sig << b.x << " " << b.y << " " << b.z << " " << b.rotation << " " << b.isSide << "\n";
    const std::string s = sig.str();
    return fnv1a(s.data(), s.size());
}

void Checkpoint::save() const
{
    if (!active)
        return;
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        f << (sliceFirst ? "tfcp2 " : "tfcp1 ") << std::hex << signature << std::dec << " " << dirIndex << " "
          << nextZ << " " << acc.totalMatches << " " << acc.listed << " "
          << acc.kernelSec << " " << (acc.truncated ? 1 : 0) << "\n";
    }
    std::rename(tmp.c_str(), path.c_str());
}

void Checkpoint::finish() const
{
    if (active)
        std::remove(path.c_str());
}

// Returns false if no usable state exists; throws on signature mismatch.
bool Checkpoint::load()
{
    std::ifstream f(path);
    if (!f)
        return false;
    std::string magic;
    std::uint64_t sig = 0;
    int trunc = 0;
    f >> magic >> std::hex >> sig >> std::dec >> dirIndex >> nextZ >>
        acc.totalMatches >> acc.listed >> acc.kernelSec >> trunc;
    if (!f || (magic != "tfcp1" && magic != "tfcp2"))
        throw std::runtime_error("checkpoint file is corrupt: " + path);
    if (sig != signature)
        throw std::runtime_error(
            "checkpoint file belongs to a different search (bounds/version/"
            "direction/formation changed): " + path + " - delete it to start over");
    sliceFirst = magic == "tfcp2";
    acc.truncated = trunc != 0;
    resumed = true;
    return true;
}
