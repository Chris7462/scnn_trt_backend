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

namespace
{

// Allocate device memory and immediately wrap it in an owning DevPtr, so
// there is no window where a raw, unmanaged pointer exists in caller scope.
internal::DevPtr cuda_malloc_dev(size_t bytes)
{
  void * p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  return internal::DevPtr(p);
}

// Same idea for pinned host memory.
internal::HostPtr cuda_malloc_host(size_t bytes)
{
  void * p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, bytes));
  return internal::HostPtr(p);
}

} // namespace

SCNNTrtBackend::Impl::~Impl() = default;

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
  preprocess_image(
    image, static_cast<float *>(buffers_.device_input.get()), config, stream_.get());

  // Run inference
  if (!context_->enqueueV3(stream_.get())) {
    throw internal::TensorRTException("Failed to enqueue inference");
  }

  // Launch GPU decode kernel directly on inference output
  internal::launch_decode_and_colorize_kernel(
    static_cast<float *>(buffers_.device_seg_output.get()),
    static_cast<float *>(buffers_.device_exist_output.get()),
    static_cast<uchar3 *>(buffers_.device_decoded_mask.get()),
    config.width, config.height,
    config.num_classes, config.num_lanes,
    config.exist_threshold,
    stream_.get()
  );

  // Async copy decoded mask to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_seg_output.get(), buffers_.device_decoded_mask.get(),
    mask_bytes_, cudaMemcpyDeviceToHost, stream_.get()));

  // Async copy existence output to pinned memory
  CUDA_CHECK(cudaMemcpyAsync(
    buffers_.pinned_exist_output.get(), buffers_.device_exist_output.get(),
    exist_output_size_, cudaMemcpyDeviceToHost, stream_.get()));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

  // Build result
  SCNNResult result;

  // Create cv::Mat from pinned memory and clone
  cv::Mat segmentation(config.height, config.width, CV_8UC3, buffers_.pinned_seg_output.get());
  result.seg_pred = segmentation.clone();

  // Copy existence probabilities (apply sigmoid since we store raw logits)
  const float * pinned_exist = static_cast<const float *>(buffers_.pinned_exist_output.get());
  for (int i = 0; i < config.num_lanes; ++i) {
    float logit = pinned_exist[i];
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

  // Allocate pinned host memory. If a later allocation throws, everything
  // already assigned above is freed automatically as the exception unwinds -
  // no manual cleanup() bookkeeping.
  buffers_.pinned_input = cuda_malloc_host(input_size_);
  buffers_.pinned_seg_output = cuda_malloc_host(mask_bytes_);
  buffers_.pinned_exist_output = cuda_malloc_host(exist_output_size_);

  // Allocate device memory
  buffers_.device_input = cuda_malloc_dev(input_size_);
  buffers_.device_seg_output = cuda_malloc_dev(seg_output_size_);
  buffers_.device_exist_output = cuda_malloc_dev(exist_output_size_);
  buffers_.device_temp_buffer = cuda_malloc_dev(input_size_);
  buffers_.device_decoded_mask = cuda_malloc_dev(mask_bytes_);

  // Set tensor addresses
  if (!context_->setTensorAddress(input_name_.c_str(), buffers_.device_input.get())) {
    throw internal::TensorRTException("Failed to set input tensor address");
  }

  if (!context_->setTensorAddress(seg_output_name_.c_str(), buffers_.device_seg_output.get())) {
    throw internal::TensorRTException("Failed to set seg_pred tensor address");
  }

  if (!context_->setTensorAddress(
      exist_output_name_.c_str(), buffers_.device_exist_output.get()))
  {
    throw internal::TensorRTException("Failed to set exist_pred tensor address");
  }
}

void SCNNTrtBackend::Impl::initialize_streams()
{
  cudaStream_t raw = nullptr;
  CUDA_CHECK(cudaStreamCreate(&raw));
  stream_.reset(raw);
}

void SCNNTrtBackend::Impl::initialize_constants()
{
  // Initialize CUDA constant memory once
  internal::initialize_mean_std_constants();
  internal::initialize_colormap_constants();
}

void SCNNTrtBackend::Impl::warmup_engine(const SCNNTrtBackend::Config & config)
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input.get(), 0, input_size_, stream_.get()));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_.get())) {
      throw internal::TensorRTException("Failed to enqueue warmup inference");
    }

    // Launch decode kernel to warm up all GPU kernels
    internal::launch_decode_and_colorize_kernel(
      static_cast<float *>(buffers_.device_seg_output.get()),
      static_cast<float *>(buffers_.device_exist_output.get()),
      static_cast<uchar3 *>(buffers_.device_decoded_mask.get()),
      config.width, config.height,
      config.num_classes, config.num_lanes,
      config.exist_threshold,
      stream_.get()
    );

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
  }

  std::cout << "Engine warmed up with " << config.warmup_iterations << " iterations" << std::endl;
}

void SCNNTrtBackend::Impl::preprocess_image(
  const cv::Mat & image, float * output, const SCNNTrtBackend::Config & config,
  cudaStream_t stream) const
{
  // Step 1: Resize image using OpenCV (on CPU)
  cv::Mat img_wrapper(config.height, config.width, CV_32FC3, buffers_.pinned_input.get());
  cv::resize(image, img_wrapper, cv::Size(config.width, config.height));

  // Step 2: Convert to float (on CPU)
  img_wrapper.convertTo(img_wrapper, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_temp_buffer.get(), img_wrapper.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch normalization kernel
  internal::launch_normalize_kernel(
    static_cast<float *>(buffers_.device_temp_buffer.get()),
    output,
    config.width, config.height,
    stream);
}

}  // namespace scnn_trt_backend
