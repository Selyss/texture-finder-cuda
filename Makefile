# Linux/macOS: `make` builds the searcher (requires CUDA), `make test` runs
# the host-side test suite (no GPU or CUDA needed), `make tools` builds the
# formation generator and oracle-diff helpers.
#
# Windows: `make windows` (edit CCBIN if your Visual Studio install differs).
#
# ARCH defaults to the GPU in the machine (`native`, CUDA 11.6+). Override for
# a specific target, e.g. `make ARCH=sm_86`.

NVCC ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH ?= native
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17 -Iinclude

CXX ?= c++
CXXFLAGS = -O2 -std=c++17 -Iinclude

SOURCES = src/main.cu src/kernel.cu src/parser.cu

CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe"

all: build/main

build/main: $(SOURCES) include/*.cuh
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $@

windows: $(SOURCES) include/*.cuh
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) $(SOURCES) -o build/main.exe

# Host-only: compiles the exact headers the kernel uses, plus the parser
# (compiled as C++), against the doctest suite.
build/test: test/test.cpp src/parser.cu include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Itest test/test.cpp -x c++ src/parser.cu -o $@

test: build/test
	./build/test

tools: build/gen_formation build/oracle_diff build/cpu_search

build/gen_formation: test/gen_formation.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/gen_formation.cpp -o $@

build/oracle_diff: test/oracle_diff.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/oracle_diff.cpp -o $@

build/cpu_search: test/cpu_search.cpp src/parser.cu include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/cpu_search.cpp -x c++ src/parser.cu -o $@

clean:
	rm -rf build

.PHONY: all windows test tools clean
