#pragma once

#include <array>
#include <charconv>
#include <optional>
#include <string_view>

#include "texture/texture_rules.hpp"

// How the versions are shown, and the ways a user may name one.

constexpr std::array<std::string_view, NUM_VERSIONS> VERSION_NAMES{"1.21.2+", "1.13 - 1.21.1", "<=1.12.2",
                                                                   "Sodium 1.0-4.1 (MC 1.16 - 1.18.2)",
                                                                   "Sodium 4.2-4.8 (MC 1.19 - 1.19.3)"};

[[nodiscard]] constexpr std::string_view version_name(Version version) noexcept {
  return VERSION_NAMES[static_cast<std::size_t>(version)];
}

namespace version_detail {

// "1.21.4" as 12104, "1.16" as 11600. Empty unless the text is two or three
// dot-separated numbers.
[[nodiscard]] inline std::optional<int> minecraft_release(std::string_view text) noexcept {
  int parts[3]{0, 0, 0};
  int count{0};
  const auto* at{text.data()};
  const auto* const end{at + text.size()};
  while (true) {
    if (count == 3)
      return std::nullopt;
    const auto [next, error]{std::from_chars(at, end, parts[count])};
    if (error != std::errc{} || parts[count] < 0 || parts[count] > 99)
      return std::nullopt;
    ++count;
    if (next == end)
      break;
    if (*next != '.')
      return std::nullopt;
    at = next + 1;
  }
  if (count < 2)
    return std::nullopt;
  return parts[0] * 10000 + parts[1] * 100 + parts[2];
}

} // namespace version_detail

// Accepts the version number 0-4, a rule set's name (modern, legacy,
// vanilla12, sodium, sodium19), a Minecraft release such as 1.21.4 or 1.12.2,
// or sodium-<release> for the two Sodium rule sets. Empty for anything else,
// including a Sodium release neither rule set covers.
[[nodiscard]] inline std::optional<Version> parse_version(std::string_view text) noexcept {
  constexpr std::array<std::string_view, NUM_VERSIONS> RULE_NAMES{"modern", "legacy", "vanilla12", "sodium",
                                                                  "sodium19"};
  for (std::size_t i{0}; i < RULE_NAMES.size(); ++i) {
    if (text == RULE_NAMES[i] || (text.size() == 1 && text.front() == static_cast<char>('0' + i)))
      return static_cast<Version>(i);
  }

  constexpr std::string_view SODIUM_PREFIX{"sodium-"};
  if (text.starts_with(SODIUM_PREFIX)) {
    const auto release{version_detail::minecraft_release(text.substr(SODIUM_PREFIX.size()))};
    if (!release)
      return std::nullopt;
    if (*release >= 11600 && *release < 11900)
      return Version::sodium;
    if (*release >= 11900 && *release <= 11903)
      return Version::sodium19;
    return std::nullopt;
  }

  const auto release{version_detail::minecraft_release(text)};
  if (!release)
    return std::nullopt;
  if (*release >= 12102)
    return Version::modern;
  return *release >= 11300 ? Version::legacy : Version::vanilla12;
}
