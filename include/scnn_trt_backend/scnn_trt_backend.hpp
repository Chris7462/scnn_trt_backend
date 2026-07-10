#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>
#include <memory>
#include <mutex>
#include <string>

// OpenCV includes
#include <opencv2/core.hpp>


namespace scnn_trt_backend
{

/**
 * @brief Result structure for SCNN inference
 */
struct SCNNResult
{
  cv::Mat seg_pred;                   // Segmentation mask (H, W, 3) colored BGR
  std::array<float, 4> exist_pred;    // Lane existence probabilities [lane1, lane2, lane3, lane4]
};

// Public log level enum, decoupled from nvinfer1::ILogger::Severity.
// Values match the existing "log_level" ROS2 parameter convention
// (0: Internal Error, 1: Error, 2: Warning, 3: Info, 4: Verbose).
enum class LogLevel
{
  kInternalError = 0,
  kError = 1,
  kWarning = 2,
  kInfo = 3,
  kVerbose = 4
};

// Optimized TensorRT inference class for SCNN lane detection
class SCNNTrtBackend
{
public:
  struct Config
  {
    /**
     * @brief Input image height
     */
    int height;

    /**
     * @brief Input image width
     */
    int width;

    /**
     * @brief Number of segmentation classes (background + 4 lanes = 5)
     */
    int num_classes;

    /**
     * @brief Number of lanes
     */
    int num_lanes;

    /**
     * @brief Lane existence threshold
     */
    float exist_threshold;

    /**
     * @brief Number of warmup iterations before timing starts
     * @details This is used to ensure that the CUDA kernels and GPU resources are properly initialized
     * and cached before actual inference timing begins. This helps to avoid cold start penalties.
     * - The first iteration initializes CUDA kernels and allocates any lazy GPU resources.
     * - The second iteration ensures everything is properly warmed up and gives more consistent timing.
     * - Set to 0 to disable warmup iterations.
     */
    int warmup_iterations;

    /**
     * @brief Log level for TensorRT messages
     * @details This controls the verbosity of TensorRT logging.
     */
    LogLevel log_level;

    /**
     * @brief Default constructor with SCNN-specific defaults
     */
    Config()
    : height(288), width(952), num_classes(5), num_lanes(4),
      exist_threshold(0.5f), warmup_iterations(2),
      log_level(LogLevel::kWarning) {}
  };

  // Constructor with configuration
  explicit SCNNTrtBackend(const std::string & engine_path, const Config & config = Config());

  // Destructor (defined in .cpp: required since Impl is incomplete here)
  ~SCNNTrtBackend();

  // Disable copy and move semantics - use std::unique_ptr for ownership transfer
  SCNNTrtBackend(const SCNNTrtBackend &) = delete;
  SCNNTrtBackend & operator=(const SCNNTrtBackend &) = delete;
  SCNNTrtBackend(SCNNTrtBackend &&) = delete;
  SCNNTrtBackend & operator=(SCNNTrtBackend &&) = delete;

  /**
   * @brief Run lane detection inference
   * @param image Input image (BGR format, CV_8UC3)
   * @return SCNNResult containing colored segmentation mask and lane existence probabilities
   */
  SCNNResult infer(const cv::Mat & image);

private:
  // Configuration (plain data - no TensorRT/CUDA types, safe to keep as a direct member)
  Config config_;

  // Opaque implementation - hides TensorRT (NvInfer.h) and CUDA (cuda_runtime.h)
  // types from consumers of this header.
  class Impl;
  std::unique_ptr<Impl> impl_;

  // Thread safety
  mutable std::mutex infer_mutex_;
};

}  // namespace scnn_trt_backend
