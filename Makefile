# Backend selection: BACKEND=cuda|metal|cpu, default auto —
#   nvcc present        -> cuda   (NVIDIA GPU)
#   else on macOS       -> metal  (Apple GPU)
#   else                -> cpu    (portable multithreaded C++)
# All backends implement the same runSearch() (include/kernel.cuh) and
# compile the same include/texture.cuh, so behavior is identical; run
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

ARCH ?= native
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17 -Iinclude

CXX ?= c++
CXXFLAGS = -O3 -std=c++17 -Iinclude

COMMON_SRC = src/main.cpp src/parser.cpp

CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe"

all: build/main

ifeq ($(BACKEND),cuda)
build/main: $(COMMON_SRC) src/kernel.cu include/*.cuh
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $(COMMON_SRC) src/kernel.cu -o $@
else ifeq ($(BACKEND),metal)
build/main: $(COMMON_SRC) src/backend_metal.mm build/tf_shader_src.h include/*.cuh
	$(CXX) $(CXXFLAGS) -Ibuild -fobjc-arc $(COMMON_SRC) src/backend_metal.mm \
	    -framework Metal -framework Foundation -o $@
else
build/main: $(COMMON_SRC) src/backend_cpu.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(COMMON_SRC) src/backend_cpu.cpp -pthread -o $@
endif

# The Metal shader source is texture.cuh + match.metal wrapped into one C++
# raw string literal, so the shader compiles the same RNG single-source.
build/tf_shader_src.h: include/texture.cuh src/match.metal
	@mkdir -p build
	{ printf '%s\n' 'R"TFMSL('; cat include/texture.cuh src/match.metal; printf '%s\n' ')TFMSL"'; } > $@

# On Windows run e.g. `make windows NVCC=nvcc` (the Unix-path default does
# not apply there); the mkdir line tolerates the directory already existing.
windows: $(COMMON_SRC) src/kernel.cu include/*.cuh
	-mkdir build
	$(NVCC) -ccbin $(CCBIN) $(NVCCFLAGS) $(COMMON_SRC) src/kernel.cu -o build/main.exe

# Host-only: compiles the exact headers the kernel uses, plus the parser,
# against the doctest suite.
build/test: test/test.cpp test/doctest.h src/parser.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Itest test/test.cpp src/parser.cpp -o $@

test: build/test
	./build/test

tools: build/gen_formation build/oracle_diff build/cpu_search

build/gen_formation: test/gen_formation.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/gen_formation.cpp -o $@

build/oracle_diff: test/oracle_diff.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/oracle_diff.cpp -o $@

build/cpu_search: test/cpu_search.cpp src/parser.cpp include/*.cuh
	@mkdir -p build
	$(CXX) $(CXXFLAGS) test/cpu_search.cpp src/parser.cpp -o $@

clean:
	rm -rf build

.PHONY: all windows test tools clean
