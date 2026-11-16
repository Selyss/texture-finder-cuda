#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "texture/formation_io.hpp"
#include "search/session.hpp"

namespace {

// Coordinates are validated against the world border plus slack; this also
// guarantees the kernel's grid-stride arithmetic cannot overflow int.
constexpr long long COORD_LIMIT = 30'001'000;
constexpr long long Y_LIMIT = 20'000;

void usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction>"
            << " [--checkpoint <state-file>]\n"
            << "  version:   0 = 1.21.2+, 1 = 1.13 - 1.21.1, 2 = <=1.12.2,\n"
            << "             3 = Sodium 1.0-4.1 (MC 1.16-1.18.2), 4 = Sodium 4.2-4.8 (MC 1.19-1.19.3)\n"
            << "  direction: 0 = North, 1 = West, 2 = South, 3 = East, or 'all' to search\n"
            << "             every orientation and report which one matched\n"
            << "  --checkpoint: persist progress after every slice; rerunning the same\n"
            << "             command resumes an interrupted search from the saved state\n";
}

int parse_int_arg(const std::string& arg, const char* name, long long lo, long long hi) {
  long long v;
  size_t pos = 0;
  try {
    v = std::stoll(arg, &pos);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(name) + " is not a number: " + arg);
  }
  if (pos != arg.size())
    throw std::runtime_error(std::string(name) + " is not a number: " + arg);
  if (v < lo || v > hi)
    throw std::runtime_error(std::string(name) + " must be in [" + std::to_string(lo) + ", " +
                             std::to_string(hi) + "], got " + arg);
  return (int)v;
}

} // namespace

int main(int argc, char* argv[]) {
  auto start_wall = std::chrono::high_resolution_clock::now();

  std::string checkpoint_path;
  if (argc == 12 && std::string(argv[10]) == "--checkpoint") {
    checkpoint_path = argv[11];
    argc = 10;
  }
  if (argc != 10) {
    usage(argv[0]);
    return 1;
  }

  SearchBounds bounds;
  int version;
  std::vector<int> directions;
  std::vector<BlockInfo> formation;
  try {
    bounds.x_min = parse_int_arg(argv[1], "x_min", -COORD_LIMIT, COORD_LIMIT);
    bounds.x_max = parse_int_arg(argv[2], "x_max", -COORD_LIMIT, COORD_LIMIT);
    bounds.y_min = parse_int_arg(argv[3], "y_min", -Y_LIMIT, Y_LIMIT);
    bounds.y_max = parse_int_arg(argv[4], "y_max", -Y_LIMIT, Y_LIMIT);
    bounds.z_min = parse_int_arg(argv[5], "z_min", -COORD_LIMIT, COORD_LIMIT);
    bounds.z_max = parse_int_arg(argv[6], "z_max", -COORD_LIMIT, COORD_LIMIT);
    if (bounds.x_min > bounds.x_max || bounds.y_min > bounds.y_max || bounds.z_min > bounds.z_max)
      throw std::runtime_error("min bound exceeds max bound");

    version = parse_int_arg(argv[7], "version", 0, NUM_VERSIONS - 1);

    if (std::string(argv[9]) == "all")
      directions = {0, 1, 2, 3};
    else
      directions = {parse_int_arg(argv[9], "direction", 0, 3)};

    formation = parse_formation_file(argv[8]);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    usage(argv[0]);
    return 1;
  }

  const auto outcome = run_search_session({bounds, formation, version, directions, checkpoint_path});
  if (outcome.exit_status != 0)
    return outcome.exit_status;

  auto end_wall = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_wall - start_wall;
  std::cout << "Kernel time: " << outcome.kernel_seconds << " seconds" << std::endl;
  std::cout << "Search completed in " << elapsed.count() << " seconds" << std::endl;
  return 0;
}
