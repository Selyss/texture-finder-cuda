#include "search/checkpoint.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "search/session.hpp"

namespace {

constexpr std::string_view DIRECTION_FIRST_MAGIC{"tfcp1"};
constexpr std::string_view SLICE_FIRST_MAGIC{"tfcp2"};

[[nodiscard]] constexpr std::uint64_t fnv1a(std::string_view text) noexcept {
  std::uint64_t hash{0xCBF29CE484222325ull};
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= 0x100000001B3ull;
  }
  return hash;
}

} // namespace

std::uint64_t checkpoint_signature(const SearchRequest& request) {
  const auto& bounds{request.bounds};
  std::ostringstream text;
  text << bounds.x_min << " " << bounds.x_max << " " << bounds.y_min << " " << bounds.y_max << " "
       << bounds.z_min << " " << bounds.z_max << " " << request.version << " ";
  for (const int direction : request.directions)
    text << direction << ",";
  text << "\n";
  for (const auto& block : request.formation)
    text << block.x << " " << block.y << " " << block.z << " " << block.rotation << " " << block.is_side
         << "\n";
  return fnv1a(text.str());
}

void Checkpoint::save() const {
  if (!active())
    return;
  // Write beside the target and rename, so a kill mid-write leaves the old file.
  const auto temporary{path + ".tmp"};
  {
    std::ofstream file{temporary, std::ios::trunc};
    file << (slice_first ? SLICE_FIRST_MAGIC : DIRECTION_FIRST_MAGIC) << " " << std::hex << signature
         << std::dec << " " << direction_index << " " << next_z << " " << totals.matches << " "
         << totals.listed << " " << totals.kernel_seconds << " " << (totals.truncated ? 1 : 0) << "\n";
  }
  std::rename(temporary.c_str(), path.c_str());
}

void Checkpoint::remove() const {
  if (active())
    std::remove(path.c_str());
}

bool Checkpoint::load() {
  std::ifstream file{path};
  if (!file)
    return false;
  std::string magic;
  std::uint64_t stored_signature{0};
  int truncated{0};
  file >> magic >> std::hex >> stored_signature >> std::dec >> direction_index >> next_z >> totals.matches >>
      totals.listed >> totals.kernel_seconds >> truncated;
  if (!file || (magic != DIRECTION_FIRST_MAGIC && magic != SLICE_FIRST_MAGIC))
    throw std::runtime_error("checkpoint file is corrupt: " + path);
  if (stored_signature != signature)
    throw std::runtime_error("checkpoint file belongs to a different search (bounds/version/"
                             "direction/formation changed): " +
                             path + " - delete it to start over");
  slice_first = magic == SLICE_FIRST_MAGIC;
  totals.truncated = truncated != 0;
  resumed = true;
  return true;
}
