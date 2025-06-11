#pragma once
#include <vector>
#include <string>
#include "blockinfo.cuh"

// TODO: make this .h and parser.cu should be .cpp, its host code and doesnt use cuda
std::vector<BlockInfo> parseFormationFile(const std::string &filename);