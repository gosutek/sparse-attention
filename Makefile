NVCC:=nvcc
CC:=clang
CXX:=clang++
ERROR_FLAGS:=-Wall -Wpointer-arith -Weffc++ -Wextra -Wconversion -Wsign-conversion -pedantic
OPT:= -O0

BUILD:=build

SOURCES=$(wildcard src/*.cpp) $(wildcard src/*.c) $(wildcard src/*.cu)

OBJECTS=$(SOURCES:src/%=$(BUILD)/%.o)
BINARY=$(BUILD)/test

CFLAGS=-g $(ERROR_FLAGS) $(OPT)
CFLAGS+=-fopenmp -mf16c -mavx2 -mfma

CUFLAGS=-g $(OPT) -lineinfo

CUFLAGS+=-std=c++20

CUFLAGS+=-Xcompiler "$(ERROR_FLAGS) -Wno-pedantic -Wno-deprecated-gpu-targets"

ifeq ($(CUARCH),)
	CUFLAGS+=-gencode arch=compute_89,code=sm_89 --threads 2
else
	CUFLAGS+=-arch=$(CUARCH)
endif

CUFLAGS += -I/opt/cuda/targets/x86_64-linux/include/

all: $(BINARY)

unit: CUFLAGS+=-Xcompiler "-D_UNIT_TESTING"
unit: $(BINARY)

bounds: CUFLAGS+=-Xcompiler "-D_GLIBCXX_DEBUG"
bounds: $(BINARY)

$(BINARY): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@

$(BUILD)/mmio.c.o: src/mmio.c
	$(CC) $< $(filter-out -Werror, $(CFLAGS)) -c -o $@

$(BUILD)/%.cpp.o: src/%.cpp
	$(NVCC) $< $(CUFLAGS) -c -o $@

$(BUILD)/%.cu.o: src/%.cu
	$(NVCC) $<  $(CUFLAGS) -c -o $@

debug:
	@echo "SOURCES=$(SOURCES)"
	@echo "OBJECTS=$(OBJECTS)"

clean:
	rm -rf $(BUILD)/*

.PHONY: all test clean
