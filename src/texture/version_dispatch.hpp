#pragma once

#include <type_traits>

#include "texture/texture_rules.hpp"

// Calls `f` with a std::integral_constant that holds `version`. Code that
// depends on the version can then be compiled once per version, with the
// version switch folded away, and chosen here at run time:
//   with_version(version, [](auto tag) { return kernel<decltype(tag)::value>; })
template <class F>
decltype(auto) with_version(Version version, F&& f) {
  switch (version) {
  case Version::legacy:
    return f(std::integral_constant<Version, Version::legacy>{});
  case Version::vanilla12:
    return f(std::integral_constant<Version, Version::vanilla12>{});
  case Version::sodium:
    return f(std::integral_constant<Version, Version::sodium>{});
  case Version::sodium19:
    return f(std::integral_constant<Version, Version::sodium19>{});
  case Version::modern:
    break;
  }
  return f(std::integral_constant<Version, Version::modern>{});
}
