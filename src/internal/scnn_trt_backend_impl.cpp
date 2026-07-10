#include <fstream>
#include <iostream>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local headers
#include "internal/scnn_trt_backend_impl.hpp"
#include "internal/cuda_check.hpp"
#include "internal/trt_logger.hpp"
#include "internal/normalize_kernel.cuh"
#include "internal/decode_and_colorize_kernel.cuh"


namespace scnn_trt_backend
{

SCNNTrtBackend::Impl::~Impl()
{
  cleanup();
}

void SCNNTrtBackend::Impl::initialize(
  const std::string & engine_path, const SCNNTrtBackend::Config & config)
{
  initialize_engine(engine_path, config);
  find_tensor_names();
  initialize_memory(config);
  initialize_streams();
  initialize_constants();
  warmup_engine(config);
}

SCNNResult SCNNTrtBackend::Impl::infer(
  const cv::Mat & image, const SCNNTrtBackend::Config & config)
{
  // Preprocess directly into GPU memory
  preprocess_image(image, buffers_.device_input, config, stream_);

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw internal::TensorRTException("Failed to enqueue inference");
  }

  // Launch GPU decode kernel directly on inference output
  internal::launch_decode_and_colorize_kernel(
    buffers_.device_seg_output,
    buffers_.device_exist_output,
    buffers_.device_decoded_mask,
    config.width, config.height,
    config.num_classes, config.num_lanes,
    config.exist_threshold,
    stream_
  );

  // Async copy decoded mask to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_seg_output, buffers_.device_decoded_mask,
    mask_bytes_, cudaMemcpyDeviceToHost, stream_));

  // Async copy existence output to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_exist_output, buffers_.device_exist_output,
    exist_output_size_, cudaMemcpyDeviceToHost, stream_));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Build result
  SCNNResult result;

  // Create cv::Mat from pinned memory and clone
  cv::Mat segmentation(config.height, config.width, CV_8UC3, buffers_.pinned_seg_output);
  result.seg_pred = segmentation.clone();

  // Copy existence probabilities (apply sigmoid since we store raw logits)
  for (int i = 0; i < config.num_lanes; ++i) {
    float logit = buffers_.pinned_exist_output[i];
    result.exist_pred[i] = 1.0f / (1.0f + std::exp(-logit));
  }

  return result;
}

void SCNNTrtBackend::Impl::initialize_engine(
  const std::string & engine_path, const SCNNTrtBackend::Config & config)
{
  // Initialize logger
  logger_ = std::make_unique<internal::Logger>(internal::to_trt_severity(config.log_level));

  auto engine_data = load_engine_file(engine_path);

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw internal::TensorRTException("Failed to create TensorRT runtime");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw internal::TensorRTException("Failed to deserialize CUDA engine");
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext());
  if (!context_) {
    throw internal::TensorRTException("Failed to create execution context");
  }
}

std::vector<uint8_t> SCNNTrtBackend::Impl::load_engine_file(
  const std::string & engine_path) const
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read engine file: " + engine_path);
  }

  return buffer;
}

void SCNNTrtBackend::Impl::find_tensor_names()
{
  bool found_input = false;
  bool found_seg_output = false;
  bool found_exist_output = false;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;
      found_input = true;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      // SCNN has two outputs: seg_pred and exist_pred
      std::string name(tensor_name);
      if (name.find("seg") != std::string::npos) {
        seg_output_name_ = tensor_name;
        found_seg_output = true;
      } else if (name.find("exist") != std::string::npos) {
        exist_output_name_ = tensor_name;
        found_exist_output = true;
      }
    }
  }

  if (!found_input) {
    throw internal::TensorRTException("Failed to find input tensor");
  }
  if (!found_seg_output) {
    throw internal::TensorRTException("Failed to find seg_pred output tensor");
  }
  if (!found_exist_output) {
    throw internal::TensorRTException("Failed to find exist_pred output tensor");
  }

  std::cout << "Found tensors - Input: " << input_name_
            << ", Seg output: " << seg_output_name_
            << ", Exist output: " << exist_output_name_ << std::endl;
}

void SCNNTrtBackend::Impl::initialize_memory(const SCNNTrtBackend::Config & config)
{
  // Calculate memory sizes
  input_size_ = 1 * 3 * config.height * config.width * sizeof(float);
  seg_output_size_ = 1 * config.num_classes * config.height * config.width * sizeof(float);
  exist_output_size_ = 1 * config.num_lanes * sizeof(float);
  mask_bytes_ = config.height * config.width * sizeof(uchar3);

  // Allocate pinned host memory
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_input, input_size_));
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_seg_output, mask_bytes_));
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_exist_output, exist_output_size_));

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&buffers_.device_input, input_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_seg_output, seg_output_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_exist_output, exist_output_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_temp_buffer, input_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_decoded_mask, mask_bytes_));

  // Set tensor addresses
  if (!context_->setTensorAddress(input_name_.c_str(),
    static_cast<void *>(buffers_.device_input)))
  {
    throw internal::TensorRTException("Failed to set input tensor address");
  }

  if (!context_->setTensorAddress(seg_output_name_.c_str(),
    static_cast<void *>(buffers_.device_seg_output)))
  {
    throw internal::TensorRTException("Failed to set seg_pred tensor address");
  }

  if (!context_->setTensorAddress(exist_output_name_.c_str(),
    static_cast<void *>(buffers_.device_exist_output)))
  {
    throw internal::TensorRTException("Failed to set exist_pred tensor address");
  }
}

void SCNNTrtBackend::Impl::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
  if (!stream_) {
    throw internal::TensorRTException("Failed to create CUDA stream");
  }
}

void SCNNTrtBackend::Impl::initialize_constants()
{
  // Initialize CUDA constant memory once
  internal::initialize_mean_std_constants();
  internal::initialize_colormap_constants();
}

void SCNNTrtBackend::Impl::warmup_engine(const SCNNTrtBackend::Config & config)
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input, 0, input_size_, stream_));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_)) {
      throw internal::TensorRTException("Failed to enqueue warmup inference");
    }

    // Launch decode kernel to warm up all GPU kernels
    internal::launch_decode_and_colorize_kernel(
      buffers_.device_seg_output,
      buffers_.device_exist_output,
      buffers_.device_decoded_mask,
      config.width, config.height,
      config.num_classes, config.num_lanes,
      config.exist_threshold,
      stream_
    );

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  std::cout << "Engine warmed up with " << config.warmup_iterations << " iterations" << std::endl;
}

void SCNNTrtBackend::Impl::cleanup() noexcept
{
  // Free pinned host memory
  if (buffers_.pinned_input) {
    cudaFreeHost(buffers_.pinned_input);
  }

  if (buffers_.pinned_seg_output) {
    cudaFreeHost(buffers_.pinned_seg_output);
  }

  if (buffers_.pinned_exist_output) {
    cudaFreeHost(buffers_.pinned_exist_output);
  }

  // Free device memory
  if (buffers_.device_input) {
    cudaFree(buffers_.device_input);
  }

  if (buffers_.device_seg_output) {
    cudaFree(buffers_.device_seg_output);
  }

  if (buffers_.device_exist_output) {
    cudaFree(buffers_.device_exist_output);
  }

  if (buffers_.device_temp_buffer) {
    cudaFree(buffers_.device_temp_buffer);
  }

  if (buffers_.device_decoded_mask) {
    cudaFree(buffers_.device_decoded_mask);
  }

  // Reset all pointers to nullptr
  buffers_ = MemoryBuffers{};

  // Destroy stream safely
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void SCNNTrtBackend::Impl::preprocess_image(
  const cv::Mat & image, float * output, const SCNNTrtBackend::Config & config,
  cudaStream_t stream) const
{
  // Step 1: Resize image using OpenCV (on CPU)
  cv::Mat img_wrapper(config.height, config.width, CV_32FC3, buffers_.pinned_input);
  cv::resize(image, img_wrapper, cv::Size(config.width, config.height));

  // Step 2: Convert to float (on CPU)
  img_wrapper.convertTo(img_wrapper, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_temp_buffer, img_wrapper.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch normalization kernel
  internal::launch_normalize_kernel(
    buffers_.device_temp_buffer,
    output,
    config.width, config.height,
    stream);
}

}  // namespace scnn_trt_backend
