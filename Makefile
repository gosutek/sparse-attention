NVCC:=nvcc
CC:=clang
CXX:=clang++

BUILD:=build

SOURCES=$(wildcard src/*.cpp) $(wildcard src/*.c) $(wildcard src/*.cu)

OBJECTS=$(SOURCES:src/%=$(BUILD)/%.o)
BINARY=$(BUILD)/test

CFLAGS=-g -Wall -Wpointer-arith -Werror -O0 -Weffc++ -Wextra -Wconversion -Wsign-conversion -pedantic-errors

CFLAGS+=-fopenmp -mf16c -mavx2 -mfma

CUFLAGS+=-g -O0 -lineinfo

ifeq ($(CUARCH),)
	CUFLAGS+=-gencode arch=compute_89,code=sm_89 --threads 2
else
	CUFLAGS+=-arch=$(CUARCH)
endif

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

$(BUILD)/mmio.c.o: src/mmio.c
	$(CC) $< $(filter-out -Werror, $(CFLAGS)) -c -o $@

$(BUILD)/%.cpp.o: src/%.cpp
	$(CXX) $< $(CFLAGS) -c -o $@

debug:
	@echo "SOURCES=$(SOURCES)"
	@echo "OBJECTS=$(OBJECTS)"

clean:
	rm -rf $(BUILD)/*

.PHONY: all clean format
