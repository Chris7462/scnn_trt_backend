#pragma once

// OpenCV includes
#include <opencv2/core.hpp>


namespace scnn_trt_backend
{

namespace utils
{

// Utility functions
cv::Mat create_overlay(const cv::Mat & original, const cv::Mat & segmentation);

}  // namespace utils

}  // namespace scnn_trt_backend
