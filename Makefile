NVCC:=nvcc
CC:=clang
CXX:=clang++
ERROR_FLAGS:=-Wall -Wpointer-arith -Weffc++ -Wextra -Wconversion -Wsign-conversion -pedantic
OPT:= -O0

BUILD_DIR:=build

SRC_DIR:=src
EXT_DIR:=extern

SOURCES:=$(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(EXT_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cu)
TEST_SOURCES:=$(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(EXT_DIR)/*.c)

OBJECTS:=$(SOURCES:$(SRC_DIR)/%=$(BUILD_DIR)/%.o)
TEST_OBJECTS:=$(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%.o,$(patsubst $(EXT_DIR)/%,$(BUILD_DIR)/%.o,$(TEST_SOURCES)))

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

all: $(BUILD_DIR)/cute

unit: CUFLAGS+=-Xcompiler "-D_UNIT_TESTING"
unit: $(BUILD_DIR)/unit_test_write_hrpb

bounds: CUFLAGS+=-Xcompiler "-D_GLIBCXX_DEBUG"
bounds: $(BUILD_DIR)/cute_bounds_checking

$(BUILD_DIR)/cute: $(OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(OBJECTS) -o $@

$(BUILD_DIR)/unit_test_write_hrpb: $(TEST_OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(TEST_OBJECTS) -o $@

$(BUILD_DIR)/cute_bounds_checking: $(OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(OBJECTS) -o $@

$(BUILD_DIR)/test.cpp.o: $(TEST_DIR)/test.cpp
	@mkdir -p $(@D)
	$(NVCC) $< $(CUFLAGS) -c -o $@

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(NVCC) $< $(CUFLAGS) -c -o $@

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $<  $(CUFLAGS) -c -o $@

$(BUILD_DIR)/%.c.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $< $(filter-out -Werror, $(CFLAGS)) -c -o $@

clean:
	rm -rf $(BUILD_DIR)/*

debug:
	@echo "TEST_SOURCES=$(TEST_SOURCES)"
	@echo "TEST_OBJECTS=$(TEST_OBJECTS)"

.PHONY: all unit bounds clean
