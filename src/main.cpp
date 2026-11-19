#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "search/checkpoint.hpp"
#include "search/session.hpp"
#include "texture/formation_io.hpp"
#include "texture/version_names.hpp"

namespace {

// The world border plus slack. Staying inside it also keeps the backends'
// index arithmetic within int.
constexpr std::int64_t COORD_LIMIT{30'001'000};
constexpr std::int64_t Y_LIMIT{20'000};

constexpr std::string_view USAGE{
    " <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction> [options]\n"
    "   or: <file> --version V --y MIN MAX (--x MIN MAX --z MIN MAX | --center X Z --radius R)\n"
    "              [--facing D] [options]\n"
    "\n"
    "  version:   a Minecraft release (1.21.4, 1.16.5, 1.12.2), sodium-<release> for\n"
    "             Sodium on 1.16 - 1.19.3, or the number of a rule set:\n"
    "             0 = 1.21.2+, 1 = 1.13 - 1.21.1, 2 = <=1.12.2,\n"
    "             3 = Sodium 1.0-4.1 (MC 1.16-1.18.2), 4 = Sodium 4.2-4.8 (MC 1.19-1.19.3)\n"
    "  direction: 0 = North, 1 = West, 2 = South, 3 = East (or the name), or 'all' to\n"
    "             search every orientation and report which one matched.\n"
    "             --facing defaults to all\n"
    "  --center X Z --radius R: search x in [X-R, X+R] and z in [Z-R, Z+R]\n"
    "\n"
    "options:\n"
    "  --checkpoint <state-file>: persist progress after every slice; rerunning the\n"
    "             same command resumes an interrupted search from the saved state\n"
    "  --devices LIST: on a CUDA build, the GPUs to use, such as 0,2. The default\n"
    "             is every GPU, and the work is shared by how fast each one is\n"
    "  --part I/N: search only piece I of N equal pieces of the z range, so N\n"
    "             machines can each take one; tools/merge_parts.py combines the\n"
    "             --json results and checks that no piece is missing\n"
    "  --quiet:   print only the 'Match found' lines, with no progress line\n"
    "  --json:    print one JSON object with the matches instead of the text report\n"
    "  --help:    show this text\n"};

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct HelpRequested {};

[[nodiscard]] std::int64_t parse_number(std::string_view text, std::string_view name, std::int64_t min,
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
  return value;
}

[[nodiscard]] int parse_int(std::string_view text, std::string_view name, std::int64_t min,
                            std::int64_t max) {
  return static_cast<int>(parse_number(text, name, min, max));
}

[[nodiscard]] Version version_from(std::string_view text) {
  if (const auto version{parse_version(text)})
    return *version;
  throw UsageError{"unknown version: " + std::string{text}};
}

[[nodiscard]] std::vector<int> directions_from(std::string_view text) {
  if (text == "all")
    return {0, 1, 2, 3};
  for (int direction{0}; direction < 4; ++direction) {
    const auto name{direction_name(direction)};
    // North, north and n all name direction 0.
    const auto matches{text.size() == name.size() || text.size() == 1};
    if (matches && !text.empty() &&
        std::equal(text.begin(), text.end(), name.begin(), [](char given, char expected) {
          return given == expected || given == expected - 'A' + 'a';
        }))
      return {direction};
  }
  return {parse_int(text, "direction", 0, 3)};
}

struct Range {
  std::string_view min, max;
};

// The command line split into options and what is left over, all still text.
struct Arguments {
  std::vector<std::string_view> positional;
  std::optional<Range> x, y, z, center;
  std::optional<std::string_view> radius, version, facing, checkpoint, devices, part;
  OutputMode output{OutputMode::text};
};

[[nodiscard]] Arguments split_arguments(std::span<const char* const> args) {
  Arguments result;
  for (std::size_t i{0}; i < args.size(); ++i) {
    const std::string_view arg{args[i]};
    // A negative coordinate starts with one dash, so only -- starts an option.
    if (!arg.starts_with("--") && arg != "-h") {
      result.positional.push_back(arg);
      continue;
    }
    const auto value{[&]() -> std::string_view {
      if (++i >= args.size())
        throw UsageError{std::string{arg} + " needs " + (arg == "--center" ? "two values" : "a value")};
      return args[i];
    }};
    const auto pair{[&]() -> Range {
      const auto first{value()};
      if (i + 1 >= args.size())
        throw UsageError{std::string{arg} + " needs two values"};
      return {first, value()};
    }};

    if (arg == "--help" || arg == "-h")
      throw HelpRequested{};
    else if (arg == "--json")
      result.output = OutputMode::json;
    else if (arg == "--quiet")
      result.output = OutputMode::quiet;
    else if (arg == "--x")
      result.x = pair();
    else if (arg == "--y")
      result.y = pair();
    else if (arg == "--z")
      result.z = pair();
    else if (arg == "--center")
      result.center = pair();
    else if (arg == "--radius")
      result.radius = value();
    else if (arg == "--version")
      result.version = value();
    else if (arg == "--facing")
      result.facing = value();
    else if (arg == "--checkpoint")
      result.checkpoint = value();
    else if (arg == "--devices")
      result.devices = value();
    else if (arg == "--part")
      result.part = value();
    else
      throw UsageError{"unknown option: " + std::string{arg}};
  }
  return result;
}

struct Fields {
  Range x, y, z;
  std::string_view version, file, facing;
};

// The nine-argument form, or the named one. Throws if they are mixed or
// something is missing.
[[nodiscard]] Fields fields_of(const Arguments& args, SearchBounds& centered) {
  const auto named{args.x || args.y || args.z || args.center || args.radius || args.version || args.facing};
  if (args.positional.size() == 9) {
    if (named)
      throw UsageError{"give the bounds, version and direction either in order or as options, not both"};
    const auto& p{args.positional};
    return {{p[0], p[1]}, {p[2], p[3]}, {p[4], p[5]}, p[6], p[7], p[8]};
  }
  if (args.positional.size() != 1)
    throw UsageError{args.positional.empty() && !named ? ""
                                                       : "expected one formation file or all nine arguments"};

  if (!args.version)
    throw UsageError{"--version is required"};
  if (!args.y)
    throw UsageError{"--y MIN MAX is required"};
  const auto by_center{args.center.has_value() || args.radius.has_value()};
  if (by_center == (args.x.has_value() || args.z.has_value()))
    throw UsageError{"give the area as --x MIN MAX --z MIN MAX, or as --center X Z --radius R"};
  if (!by_center) {
    if (!args.x || !args.z)
      throw UsageError{"--x and --z are both required"};
    return {*args.x, *args.y, *args.z, *args.version, args.positional.front(), args.facing.value_or("all")};
  }
  if (!args.center || !args.radius)
    throw UsageError{"--center and --radius are both required"};

  // Clamped to the limit, so a wide radius near the border is still a search.
  const auto x{parse_number(args.center->min, "center x", -COORD_LIMIT, COORD_LIMIT)};
  const auto z{parse_number(args.center->max, "center z", -COORD_LIMIT, COORD_LIMIT)};
  const auto radius{parse_number(*args.radius, "radius", 0, 2 * COORD_LIMIT)};
  const auto clamp{
      [](std::int64_t value) { return static_cast<int>(std::clamp(value, -COORD_LIMIT, COORD_LIMIT)); }};
  centered = {.x_min = clamp(x - radius),
              .x_max = clamp(x + radius),
              .y_min = 0,
              .y_max = 0,
              .z_min = clamp(z - radius),
              .z_max = clamp(z + radius)};
  return {{}, *args.y, {}, *args.version, args.positional.front(), args.facing.value_or("all")};
}

// "0,2" as {0, 2}.
[[nodiscard]] std::vector<int> device_list(std::string_view text) {
  std::vector<int> devices;
  for (std::size_t start{0}; start <= text.size();) {
    const auto comma{std::min(text.find(',', start), text.size())};
    devices.push_back(parse_int(text.substr(start, comma - start), "device", 0, 1023));
    start = comma + 1;
  }
  return devices;
}

// Which piece of a search this run is. `search_id` identifies the whole
// search, so results of different searches are never merged.
struct Part {
  int index{1}, count{1};
  std::uint64_t search_id{0};
};

struct Command {
  SearchRequest request;
  Part part;
};

// Piece `index` of `count` equal runs of z rows: every row is in exactly one.
[[nodiscard]] SearchBounds part_of(SearchBounds bounds, std::string_view text, Part& part) {
  const auto slash{text.find('/')};
  if (slash == std::string_view::npos)
    throw UsageError{"--part takes I/N, such as 3/16"};
  const auto rows{static_cast<std::int64_t>(bounds.nz())};
  part.count = parse_int(text.substr(slash + 1), "part count", 1, rows);
  part.index = parse_int(text.substr(0, slash), "part", 1, part.count);
  const auto first{rows * (part.index - 1) / part.count};
  const auto end{rows * part.index / part.count};
  const auto z_min{std::int64_t{bounds.z_min}};
  bounds.z_min = static_cast<int>(z_min + first);
  bounds.z_max = static_cast<int>(z_min + end - 1);
  return bounds;
}

// args excludes the program name.
[[nodiscard]] Command parse_request(std::span<const char* const> args) {
  const auto arguments{split_arguments(args)};
  try {
    SearchBounds bounds{};
    const auto fields{fields_of(arguments, bounds)};
    if (!arguments.center) {
      bounds.x_min = parse_int(fields.x.min, "x_min", -COORD_LIMIT, COORD_LIMIT);
      bounds.x_max = parse_int(fields.x.max, "x_max", -COORD_LIMIT, COORD_LIMIT);
      bounds.z_min = parse_int(fields.z.min, "z_min", -COORD_LIMIT, COORD_LIMIT);
      bounds.z_max = parse_int(fields.z.max, "z_max", -COORD_LIMIT, COORD_LIMIT);
    }
    bounds.y_min = parse_int(fields.y.min, "y_min", -Y_LIMIT, Y_LIMIT);
    bounds.y_max = parse_int(fields.y.max, "y_max", -Y_LIMIT, Y_LIMIT);
    if (bounds.x_min > bounds.x_max || bounds.y_min > bounds.y_max || bounds.z_min > bounds.z_max)
      throw UsageError{"min bound exceeds max bound"};

    if (arguments.devices)
      choose_search_devices(device_list(*arguments.devices));
    const auto version{version_from(fields.version)};
    auto directions{directions_from(fields.facing)};
    auto formation{parse_formation_file(std::string{fields.file})};
    if (arguments.output == OutputMode::text)
      std::cout << formation.size() << " blocks read from formation file: " << fields.file << std::endl;
    SearchRequest request{.bounds = bounds,
                          .formation = std::move(formation),
                          .version = version,
                          .directions = std::move(directions),
                          .checkpoint_path = std::string{arguments.checkpoint.value_or("")},
                          .output = arguments.output};
    Part part{.index = 1, .count = 1, .search_id = checkpoint_signature(request)};
    if (arguments.part) {
      request.bounds = part_of(bounds, *arguments.part, part);
      if (arguments.output == OutputMode::text)
        std::cout << "Part: " << part.index << " of " << part.count << " (z " << request.bounds.z_min
                  << " to " << request.bounds.z_max << ")" << std::endl;
    }
    return {std::move(request), part};
  } catch (const UsageError&) {
    throw;
  } catch (const std::exception& error) {
    throw UsageError{error.what()}; // a bad formation file also gets the usage text
  }
}

// One object on one line. Every string in it is one of ours, so none needs escaping.
void print_json(const Command& command, const SearchOutcome& outcome, double seconds) {
  const auto& request{command.request};
  const auto& bounds{request.bounds};
  const auto& part{command.part};
  char search_id[17];
  std::snprintf(search_id, sizeof search_id, "%016llx", static_cast<unsigned long long>(part.search_id));
  std::cout << "{\"status\":\"" << (outcome.status == SessionStatus::finished ? "finished" : "stopped")
            << "\",\"backend\":\"" << search_backend_name() << "\",\"devices\":\"" << search_hardware()
            << "\",\"version\":\"" << version_name(request.version) << "\",\"search\":\"" << outcome.search
            << "\",\"bounds\":{\"x\":[" << bounds.x_min << "," << bounds.x_max << "],\"y\":[" << bounds.y_min
            << "," << bounds.y_max << "],\"z\":[" << bounds.z_min << "," << bounds.z_max << "]},\"part\":["
            << part.index << "," << part.count << "],\"search_id\":\"" << search_id << "\",\"facing\":\""
            << (request.directions.size() > 1 ? "all" : direction_name(request.directions.front()))
            << "\",\"total_matches\":" << outcome.total_matches
            << ",\"truncated\":" << (outcome.truncated ? "true" : "false") << ",\"matches\":[";
  for (std::size_t i{0}; i < outcome.matches.size(); ++i) {
    const auto& [match, direction]{outcome.matches[i]};
    std::cout << (i ? "," : "") << "{\"x\":" << match.x << ",\"y\":" << match.y << ",\"z\":" << match.z
              << ",\"facing\":\"" << direction_name(direction) << "\"}";
  }
  std::cout << "],\"kernel_seconds\":" << outcome.kernel_seconds << ",\"seconds\":" << seconds << "}"
            << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
  const auto started{std::chrono::steady_clock::now()};
  try {
    const auto command{parse_request({argv + 1, argv + argc})};
    const auto& request{command.request};
    const auto outcome{run_search_session(request)};
    const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - started};

    if (request.output == OutputMode::json)
      print_json(command, outcome, elapsed.count());
    if (outcome.status == SessionStatus::stopped)
      return 3;
    if (request.output == OutputMode::text)
      std::cout << "Kernel time: " << outcome.kernel_seconds << " seconds\n"
                << "Search completed in " << elapsed.count() << " seconds" << std::endl;
    return 0;
  } catch (const HelpRequested&) {
    std::cout << "Usage: " << argv[0] << USAGE;
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
