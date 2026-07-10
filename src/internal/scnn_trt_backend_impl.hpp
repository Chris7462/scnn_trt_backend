#pragma once

// C++ standard library includes
#include <memory>
#include <string>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/core.hpp>

// local headers
#include "scnn_trt_backend/scnn_trt_backend.hpp"
#include "internal/trt_logger.hpp"

// NOTE: This is a private implementation header. It lives under src/internal/
// and is never installed - it must not be included by any consumer of the
// scnn_trt_backend public API. It completes the SCNNTrtBackend::Impl type that
// the public header only forward-declares.

namespace scnn_trt_backend
{

// SCNNTrtBackend::Impl - holds everything that needs NvInfer.h / cuda_runtime.h.
// Public surface is intentionally narrow: initialize() and infer() are the
// only two operations SCNNTrtBackend needs to drive. Everything else -
// engine setup, memory allocation, warmup, cleanup - is an internal
// implementation detail and stays private.
class SCNNTrtBackend::Impl
{
public:
  ~Impl();

  void initialize(const std::string & engine_path, const SCNNTrtBackend::Config & config);
  SCNNResult infer(const cv::Mat & image, const SCNNTrtBackend::Config & config);

private:
  // Initialization steps
  void initialize_engine(const std::string & engine_path, const SCNNTrtBackend::Config & config);
  void find_tensor_names();
  void initialize_memory(const SCNNTrtBackend::Config & config);
  void initialize_streams();
  void initialize_constants();
  void warmup_engine(const SCNNTrtBackend::Config & config);

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void preprocess_image(
    const cv::Mat & image, float * output, const SCNNTrtBackend::Config & config,
    cudaStream_t stream) const;

private:
  // TensorRT objects
  std::unique_ptr<internal::Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Tensor information (SCNN has 1 input and 2 outputs)
  std::string input_name_;
  std::string seg_output_name_;
  std::string exist_output_name_;

  // Memory sizes
  size_t input_size_ = 0;
  size_t seg_output_size_ = 0;
  size_t exist_output_size_ = 0;
  size_t mask_bytes_ = 0;

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

  // CUDA stream for pipelining
  cudaStream_t stream_ = nullptr;
};

}  // namespace scnn_trt_backend
