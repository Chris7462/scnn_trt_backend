// OpenCV includes
#include <opencv2/imgproc.hpp>

// Local header files
#include "scnn_trt_backend/lane_utils.hpp"


namespace scnn_trt_backend
{

namespace utils
{

cv::Mat create_overlay(const cv::Mat & original, const cv::Mat & segmentation)
{
  cv::Mat overlay = original.clone();
  cv::Mat seg_resized;

  // Resize segmentation to match original image size
  cv::resize(segmentation, seg_resized, original.size(), 0, 0, cv::INTER_NEAREST);

  // Create mask where segmentation is non-zero (detected lanes)
  cv::Mat gray;
  cv::cvtColor(seg_resized, gray, cv::COLOR_BGR2GRAY);
  cv::Mat mask = gray > 0;

  // Copy lane pixels onto original image
  seg_resized.copyTo(overlay, mask);

  return overlay;
}

}  // namespace utils

}  // namespace scnn_trt_backend
