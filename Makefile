NVCC:=nvcc
CC:=clang
CXX:=clang++
ERROR_FLAGS:=-Wall -Wpointer-arith -Weffc++ -Wextra -Wconversion -Wsign-conversion -pedantic
OPT:= -O0

BUILD_DIR:=build

SRC_DIR:=src
EXT_DIR:=extern

SOURCES:=$(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)

OBJECTS:=$(SOURCES:$(SRC_DIR)/%=$(BUILD_DIR)/%.o)

CFLAGS=-g $(ERROR_FLAGS) $(OPT)
CFLAGS+=-fopenmp -mf16c -mavx2 -mfma
CFLAGS+=-I/opt/cuda/targets/x86_64-linux/include/

CUFLAGS=-g $(OPT) -lineinfo
CUFLAGS+=-std=c++20
CUFLAGS+=-Xcompiler "$(ERROR_FLAGS) -Wno-pedantic"

ifeq ($(CUARCH),)
	CUFLAGS+=-gencode arch=compute_89,code=sm_89 --threads 2
else
	CUFLAGS+=-arch=$(CUARCH)
endif

CUFLAGS+=-I/opt/cuda/targets/x86_64-linux/include/
CUFLAGS+=-Wno-deprecated-gpu-targets

CUFLAGS+=-Xcompiler "-DMAT_SIZE=512"

all: $(BUILD_DIR)/cute

bounds: CUFLAGS+=-Xcompiler "-D_GLIBCXX_DEBUG"
bounds: $(BUILD_DIR)/cute_bounds_checking

$(BUILD_DIR)/cute: $(OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(CUFLAGS) $(OBJECTS) -o $@

$(BUILD_DIR)/cute_bounds_checking: $(OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(OBJECTS) -o $@

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $< $(CFLAGS) -c -o $@

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $< $(CUFLAGS) -c -o $@

$(BUILD_DIR)/%.c.o: $(EXT_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $< $(filter-out -Werror, $(CFLAGS)) -c -o $@

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all unit bounds clean
