// local headers
#include "scnn_trt_backend/scnn_trt_backend.hpp"
#include "internal/scnn_trt_backend_impl.hpp"
#include "internal/cuda_check.hpp"


namespace scnn_trt_backend
{

SCNNTrtBackend::SCNNTrtBackend(const std::string & engine_path, const Config & config)
: config_(config), impl_(std::make_unique<SCNNTrtBackend::Impl>())
{
  try {
    impl_->initialize(engine_path, config_);
  } catch (const std::exception & e) {
    impl_.reset();
    throw internal::TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

SCNNTrtBackend::~SCNNTrtBackend() = default;

SCNNResult SCNNTrtBackend::infer(const cv::Mat & image)
{
  std::lock_guard<std::mutex> lock(infer_mutex_);
  return impl_->infer(image, config_);
}

}  // namespace scnn_trt_backend
