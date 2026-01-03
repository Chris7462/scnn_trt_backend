#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>
#include <cstdint>


namespace config
{

// Model input size (must match the exported ONNX/TensorRT engine)
// 288x952 preserves KITTI aspect ratio (370x1226 -> 288x952, divisible by 8)
constexpr int MODEL_HEIGHT = 288;
constexpr int MODEL_WIDTH = 952;

// Number of segmentation classes (background + 4 lanes)
constexpr int NUM_CLASSES = 5;

// Number of lanes
constexpr int NUM_LANES = 4;

// Lane existence threshold (probability > threshold to consider lane exists)
constexpr float EXIST_THRESHOLD = 0.75f;

// ImageNet normalization constants
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

// Lane colors for visualization (RGB format for CUDA kernel)
// Index 0: Background (black)
// Index 1-4: Lane 1-4
constexpr std::array<std::array<uint8_t, 3>, 5> LANE_COLORMAP = {{
  {0, 0, 0},        // Background: Black
  {255, 125, 0},    // Lane 1: Orange
  {0, 255, 0},      // Lane 2: Green
  {255, 0, 0},      // Lane 3: Red
  {255, 255, 0},    // Lane 4: Yellow
}};

}  // namespace config
