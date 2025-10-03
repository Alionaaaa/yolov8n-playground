# Paths to dependencies
ONNX_ROOT = $(PWD)/third_party/onnxruntime
ONNX_INCLUDE = $(ONNX_ROOT)/include
ONNX_LIB = $(ONNX_ROOT)/lib

PYBIND11_INCLUDE = $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDE = $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")

OPENCV_INCLUDE = /usr/include/opencv4
OPENCV_LIB = /usr/lib

# Build targets
TARGET = build/yolo_onnx.so
PYTHON_DIR = python

# Source files
SRC = src/yolo_engine.cpp src/pybind_wrapper.cpp

# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -Wall -I$(ONNX_INCLUDE) -I$(OPENCV_INCLUDE) $(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE)
LDFLAGS = -shared -L$(ONNX_LIB) -lonnxruntime -lopencv_dnn -lopencv_imgproc -lopencv_core -lopencv_highgui

# Build directory
BUILD_DIR = build

all: $(BUILD_DIR) $(TARGET) install

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

install: $(TARGET)
	@echo "Installing module to Python directory..."
	cp $(TARGET) $(PYTHON_DIR)/
	@echo "Module ready in $(PYTHON_DIR)/"

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PYTHON_DIR)/yolo_onnx.so

.PHONY: all clean install