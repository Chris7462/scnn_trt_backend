#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>
#include <memory>
#include <string>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

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

// TensorRT Logger with configurable severity
class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(Severity min_severity = Severity::kWARNING)
  : min_severity_(min_severity) {}

  void log(Severity severity, const char * msg) noexcept override;

private:
  Severity min_severity_;
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
     */
    int warmup_iterations;

    /**
     * @brief Log level for TensorRT messages
     */
    Logger::Severity log_level;

    /**
     * @brief Default constructor with SCNN-specific defaults
     */
    Config()
    : height(288), width(952), num_classes(5), num_lanes(4),
      exist_threshold(0.5f), warmup_iterations(2),
      log_level(Logger::Severity::kWARNING) {}
  };

  // Constructor with configuration
  explicit SCNNTrtBackend(const std::string & engine_path, const Config & config = Config());

  // Destructor
  ~SCNNTrtBackend();

  // Disable copy and move semantics
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
  // Initialization methods
  void initialize_engine(const std::string & engine_path);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();
  void initialize_constants();
  void warmup_engine();

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void preprocess_image(const cv::Mat & image, float * output, cudaStream_t stream) const;

private:
  // Configuration
  Config config_;

  // TensorRT objects
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Tensor information (SCNN has 1 input and 2 outputs)
  std::string input_name_;
  std::string seg_output_name_;
  std::string exist_output_name_;

  // Memory sizes
  size_t input_size_;
  size_t seg_output_size_;
  size_t exist_output_size_;
  size_t mask_bytes_;

  // Memory buffers
  struct MemoryBuffers
  {
    // Pinned host memory
    float * pinned_input;
    uchar3 * pinned_seg_output;
    float * pinned_exist_output;

    // Device memory
    float * device_input;           // TensorRT engine input
    float * device_seg_output;      // TensorRT seg_pred output [1, 5, H, W]
    float * device_exist_output;    // TensorRT exist_pred output [1, 4]
    float * device_temp_buffer;     // For image preprocessing
    uchar3 * device_decoded_mask;   // Decoded segmentation mask

    MemoryBuffers()
    : pinned_input(nullptr), pinned_seg_output(nullptr), pinned_exist_output(nullptr),
      device_input(nullptr), device_seg_output(nullptr), device_exist_output(nullptr),
      device_temp_buffer(nullptr), device_decoded_mask(nullptr) {}
  } buffers_;

  // CUDA stream
  cudaStream_t stream_;
};

}  // namespace scnn_trt_backend
