#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "search/session.hpp"
#include "texture/formation_io.hpp"

namespace {

// The world border plus slack. Staying inside it also keeps the backends'
// index arithmetic within int.
constexpr std::int64_t COORD_LIMIT{30'001'000};
constexpr std::int64_t Y_LIMIT{20'000};

constexpr std::string_view USAGE{
    " <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction>"
    " [--checkpoint <state-file>]\n"
    "  version:   0 = 1.21.2+, 1 = 1.13 - 1.21.1, 2 = <=1.12.2,\n"
    "             3 = Sodium 1.0-4.1 (MC 1.16-1.18.2), 4 = Sodium 4.2-4.8 (MC 1.19-1.19.3)\n"
    "  direction: 0 = North, 1 = West, 2 = South, 3 = East, or 'all' to search\n"
    "             every orientation and report which one matched\n"
    "  --checkpoint: persist progress after every slice; rerunning the same\n"
    "             command resumes an interrupted search from the saved state\n"};

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[nodiscard]] int parse_int(std::string_view text, std::string_view name, std::int64_t min,
                            std::int64_t max) {
  std::int64_t value{};
  const auto digits{text.starts_with('+') ? text.substr(1) : text}; // from_chars rejects a plus sign
  const auto* const end{digits.data() + digits.size()};
  const auto [parsed_to, error]{std::from_chars(digits.data(), end, value)};
  if (error != std::errc{} || parsed_to != end)
    throw UsageError{std::string{name} + " is not a number: " + std::string{text}};
  if (value < min || value > max)
    throw UsageError{std::string{name} + " must be in [" + std::to_string(min) + ", " + std::to_string(max) +
                     "], got " + std::string{text}};
  return static_cast<int>(value);
}

// args excludes the program name.
[[nodiscard]] SearchRequest parse_request(std::span<const char* const> args) {
  std::string checkpoint_path;
  if (args.size() == 11 && std::string_view{args[9]} == "--checkpoint") {
    checkpoint_path = args[10];
    args = args.first(9);
  }
  if (args.size() != 9)
    throw UsageError{""}; // wrong argument count: usage text only

  try {
    const SearchBounds bounds{.x_min = parse_int(args[0], "x_min", -COORD_LIMIT, COORD_LIMIT),
                              .x_max = parse_int(args[1], "x_max", -COORD_LIMIT, COORD_LIMIT),
                              .y_min = parse_int(args[2], "y_min", -Y_LIMIT, Y_LIMIT),
                              .y_max = parse_int(args[3], "y_max", -Y_LIMIT, Y_LIMIT),
                              .z_min = parse_int(args[4], "z_min", -COORD_LIMIT, COORD_LIMIT),
                              .z_max = parse_int(args[5], "z_max", -COORD_LIMIT, COORD_LIMIT)};
    if (bounds.x_min > bounds.x_max || bounds.y_min > bounds.y_max || bounds.z_min > bounds.z_max)
      throw UsageError{"min bound exceeds max bound"};

    const auto version{parse_int(args[6], "version", 0, NUM_VERSIONS - 1)};
    auto directions{std::string_view{args[8]} == "all" ? std::vector{0, 1, 2, 3}
                                                       : std::vector{parse_int(args[8], "direction", 0, 3)}};
    auto formation{parse_formation_file(args[7])};
    std::cout << formation.size() << " blocks read from formation file: " << args[7] << std::endl;
    return {.bounds = bounds,
            .formation = std::move(formation),
            .version = version,
            .directions = std::move(directions),
            .checkpoint_path = std::move(checkpoint_path)};
  } catch (const UsageError&) {
    throw;
  } catch (const std::exception& error) {
    throw UsageError{error.what()}; // a bad formation file also gets the usage text
  }
}

} // namespace

int main(int argc, char* argv[]) {
  const auto started{std::chrono::steady_clock::now()};
  try {
    const auto request{parse_request({argv + 1, argv + argc})};
    const auto outcome{run_search_session(request)};
    if (outcome.status == SessionStatus::stopped)
      return 3;

    const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - started};
    std::cout << "Kernel time: " << outcome.kernel_seconds << " seconds\n"
              << "Search completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
  } catch (const UsageError& error) {
    if (*error.what() != '\0')
      std::cerr << "Error: " << error.what() << std::endl;
    std::cerr << "Usage: " << argv[0] << USAGE;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << std::endl;
    return 1;
  }
}
