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
#include "internal/cuda_raii.hpp"

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

  // Memory buffers. Ownership via RAII smart pointers (see cuda_raii.hpp) -
  // no manual cleanup() bookkeeping needed; destruction order below (and
  // stream_ declared last) ensures buffers_ is torn down before the stream
  // that operations on it were queued against.
  struct MemoryBuffers
  {
    // Pinned host memory
    internal::HostPtr pinned_input;
    internal::HostPtr pinned_seg_output;
    internal::HostPtr pinned_exist_output;

    // Device memory
    internal::DevPtr device_input;         // TensorRT engine input
    internal::DevPtr device_seg_output;    // TensorRT seg_pred output [1, 5, H, W]
    internal::DevPtr device_exist_output;  // TensorRT exist_pred output [1, 4]
    internal::DevPtr device_temp_buffer;   // For image preprocessing
    internal::DevPtr device_decoded_mask;  // Decoded segmentation mask
  } buffers_;

  // CUDA stream for pipelining
  internal::StreamPtr stream_;
};

}  // namespace scnn_trt_backend
