#pragma once
#include <string>
#include <vector>
#include "texture/formation.hpp"

// Parses a formation file ("x y z rotation isSide" per line; '#' comments and
// blank lines allowed). Throws std::runtime_error with file:line context on
// any invalid input. Host-only code.
std::vector<BlockInfo> parseFormationFile(const std::string &filename);
