# Backend selection: BACKEND=cuda|metal|cpu, default auto —
#   nvcc present        -> cuda   (NVIDIA GPU)
#   else on macOS       -> metal  (Apple GPU)
#   else                -> cpu    (portable multithreaded C++)
# All backends implement the same run_search() (src/search/backend.hpp) and
# compile the same src/texture/texture_rules.hpp, so behavior is identical; run
# test/e2e.sh to validate whichever backend was built.
#
# `make test` runs the host-side unit suite (no GPU needed anywhere).
# Windows: `make windows NVCC=nvcc` (edit CCBIN if your VS install differs).
# ARCH (CUDA only) defaults to the GPU in the machine (`native`, CUDA 11.6+).

UNAME_S := $(shell uname -s)
NVCC ?= $(shell command -v nvcc 2>/dev/null || { [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc; } || true)

BACKEND ?= auto
ifeq ($(BACKEND),auto)
  ifneq ($(NVCC),)
    BACKEND := cuda
  else ifeq ($(UNAME_S),Darwin)
    BACKEND := metal
  else
    BACKEND := cpu
  endif
endif

# -Wall -Wextra -Wshadow catch the class of silent integer/shadowing bugs the
# original code shipped; not -Wconversion/-Wsign-conversion, which would flag
# the deliberate unsigned wrapping arithmetic on every line.
WARNINGS = -Wall -Wextra -Wshadow

ARCH ?= native
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++20 -Isrc -Xcompiler=-Wall,-Wextra,-Wshadow

CXX ?= c++
CXXFLAGS = -O3 -std=c++20 -Isrc $(WARNINGS)

COMMON_SRC = src/main.cpp src/search/session.cpp src/search/checkpoint.cpp src/search/progress.cpp src/texture/formation_io.cpp
COMMON_HEADERS = $(wildcard src/search/*.hpp src/texture/*.hpp)
CUDA_SRC = src/backends/cuda/search.cu src/backends/cuda/bitplane_search.cu src/backends/cuda/bitplane_plan.cpp
CUDA_HEADERS = $(wildcard src/backends/cuda/*.cuh src/backends/cuda/*.hpp)

CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe"

all: build/main

# build/main is one file for every backend, so switching BACKEND must rebuild it.
BACKEND_STAMP = build/.backend-$(BACKEND)
$(BACKEND_STAMP):
	@mkdir -p build
	@rm -f build/.backend-* build/main
	@touch $@

ifeq ($(BACKEND),cuda)
build/main: $(COMMON_SRC) $(CUDA_SRC) $(COMMON_HEADERS) $(CUDA_HEADERS) $(BACKEND_STAMP)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(COMMON_SRC) $(CUDA_SRC) -o $@
else ifeq ($(BACKEND),metal)
build/main: $(COMMON_SRC) src/backends/metal/search.mm build/tf_shader_src.h $(COMMON_HEADERS) $(BACKEND_STAMP)
	$(CXX) $(CXXFLAGS) -Ibuild -fobjc-arc $(COMMON_SRC) src/backends/metal/search.mm \
	    -framework Metal -framework Foundation -o $@
else
build/main: $(COMMON_SRC) src/backends/cpu/search.cpp $(COMMON_HEADERS) $(BACKEND_STAMP)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(COMMON_SRC) src/backends/cpu/search.cpp -pthread -o $@
endif

# The Metal shader source is texture_rules.hpp + match.metal wrapped into one C++
# raw string literal, so the shader compiles the same RNG single-source.
build/tf_shader_src.h: src/texture/texture_rules.hpp src/backends/metal/match.metal
	@mkdir -p build
	{ printf '%s\n' 'R"TFMSL('; cat src/texture/texture_rules.hpp src/backends/metal/match.metal; printf '%s\n' ')TFMSL"'; } > $@

# On Windows run e.g. `make windows NVCC=nvcc` (the Unix-path default does
# not apply there); the mkdir line tolerates the directory already existing.
windows: $(COMMON_SRC) $(CUDA_SRC) $(COMMON_HEADERS) $(CUDA_HEADERS)
	-mkdir build
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) $(COMMON_SRC) $(CUDA_SRC) -o build/main.exe

# Host-only: compiles the exact headers the kernel uses, plus the parser,
# against the doctest suite.
TEST_SRC = test/test.cpp src/texture/formation_io.cpp src/backends/cuda/bitplane_plan.cpp
build/test: $(TEST_SRC) test/doctest.h $(COMMON_HEADERS) $(CUDA_HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Itest $(TEST_SRC) -o $@

test: build/test
	./build/test

tools: build/gen_formation build/oracle_diff build/cpu_search

build/gen_formation: test/gen_formation.cpp $(COMMON_HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/gen_formation.cpp -o $@

build/oracle_diff: test/oracle_diff.cpp $(COMMON_HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/oracle_diff.cpp -o $@

build/cpu_search: test/cpu_search.cpp src/texture/formation_io.cpp $(COMMON_HEADERS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/cpu_search.cpp src/texture/formation_io.cpp -o $@

clean:
	rm -rf build

.PHONY: all windows test tools clean

# Standalone fixed-stride CUDA bitplane experiment.
build/flat_cuda: experiments/rng-reduction/flat_cuda.cu experiments/rng-reduction/diagnostic_kernels.cuh $(CUDA_SRC) src/texture/formation_io.cpp $(COMMON_HEADERS) $(CUDA_HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-pthread $< $(CUDA_SRC) src/texture/formation_io.cpp -o $@
