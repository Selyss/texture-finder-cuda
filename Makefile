# Edit the path to cl.exe if your Visual Studio install is different
CCBIN="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe"
NVCC=nvcc
CFLAGS=-O2 -std=c++17

SOURCES=blockinfo.cu main.cu kernel.cu rotation.cu parser.cu
TARGET=main.exe

all:
	$(NVCC) -ccbin $(CCBIN) $(SOURCES) -o $(TARGET) $(CFLAGS)