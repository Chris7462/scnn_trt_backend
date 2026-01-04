// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#define private public
#include "scnn_trt_backend/scnn_trt_backend.hpp"
#undef private
#include "scnn_trt_backend/lane_utils.hpp"


class SCNNTrtBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Configure the detector
    scnn_trt_backend::SCNNTrtBackend::Config conf;
    conf.height = input_height_;
    conf.width = input_width_;
    conf.num_classes = num_classes_;
    conf.num_lanes = num_lanes_;
    conf.exist_threshold = exist_threshold_;
    conf.warmup_iterations = 2;
    conf.log_level = scnn_trt_backend::Logger::Severity::kINFO;

    try {
      detector_ = std::make_unique<scnn_trt_backend::SCNNTrtBackend>(engine_path_, conf);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize TensorRT detector: " << e.what();
    }
  }

  void TearDown() override
  {
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  void save_results(
    const cv::Mat & original, const cv::Mat & segmentation,
    const cv::Mat & overlay, const std::string & suffix = "")
  {
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_segmentation" + suffix + ".png", segmentation);
    cv::imwrite("test_output_overlay" + suffix + ".png", overlay);
  }

  void print_exist_pred(const std::array<float, 4> & exist_pred)
  {
    std::cout << "Lane existence probabilities: [";
    for (size_t i = 0; i < exist_pred.size(); ++i) {
      std::cout << exist_pred[i];
      if (i < exist_pred.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  std::unique_ptr<scnn_trt_backend::SCNNTrtBackend> detector_;

public:
  const int input_width_ = 952;
  const int input_height_ = 288;
  const int num_classes_ = 5;
  const int num_lanes_ = 4;
  const float exist_threshold_ = 0.5f;

private:
  const std::string engine_path_ = "scnn_vgg16_288x952.engine";
  const std::string image_path_ = "image_000.png";
};

TEST_F(SCNNTrtBackendTest, TestBasicInference)
{
  cv::Mat image = load_test_image();
  EXPECT_FALSE(image.empty());
  EXPECT_EQ(image.type(), CV_8UC3);

  std::cout << "Input image size: " << image.cols << "x" << image.rows << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  scnn_trt_backend::SCNNResult result = detector_->infer(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "GPU infer with decode: " << duration.count() << " ms" << std::endl;

  // Validate segmentation output
  EXPECT_EQ(result.seg_pred.rows, detector_->config_.height);
  EXPECT_EQ(result.seg_pred.cols, detector_->config_.width);
  EXPECT_EQ(result.seg_pred.type(), CV_8UC3);

  // Validate existence output
  print_exist_pred(result.exist_pred);
  for (size_t i = 0; i < result.exist_pred.size(); ++i) {
    EXPECT_GE(result.exist_pred[i], 0.0f) << "Existence probability should be >= 0";
    EXPECT_LE(result.exist_pred[i], 1.0f) << "Existence probability should be <= 1";
  }

  // Create overlay
  cv::Mat overlay = scnn_trt_backend::utils::create_overlay(image, result.seg_pred, 0.5f);
  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, result.seg_pred, overlay, "_gpu_optimized");
}

TEST_F(SCNNTrtBackendTest, TestMultipleInferences)
{
  cv::Mat image = load_test_image();

  const int num_iterations = 10;
  std::vector<double> inference_times;

  for (int i = 0; i < num_iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = detector_->infer(image);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(end - start);
    inference_times.push_back(duration.count());

    // Validate output consistency
    EXPECT_EQ(result.seg_pred.rows, detector_->config_.height);
    EXPECT_EQ(result.seg_pred.cols, detector_->config_.width);

    // Validate existence probabilities are valid
    for (size_t j = 0; j < result.exist_pred.size(); ++j) {
      EXPECT_GE(result.exist_pred[j], 0.0f);
      EXPECT_LE(result.exist_pred[j], 1.0f);
    }
  }

  // Calculate statistics
  double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
    inference_times.size();
  double min_time = *std::min_element(inference_times.begin(), inference_times.end());
  double max_time = *std::max_element(inference_times.begin(), inference_times.end());

  std::cout << "Multiple inference statistics:" << std::endl;
  std::cout << "  Average: " << avg_time << " ms" << std::endl;
  std::cout << "  Min: " << min_time << " ms" << std::endl;
  std::cout << "  Max: " << max_time << " ms" << std::endl;

  // Performance expectations (adjust based on your hardware)
  EXPECT_LT(avg_time, 100.0);  // Should be less than 100ms on decent hardware
}

TEST_F(SCNNTrtBackendTest, TestBenchmarkInference)
{
  cv::Mat image = load_test_image();

  const int warmup_iterations = 10;
  const int benchmark_iterations = 100;

  // Warmup
  for (int i = 0; i < warmup_iterations; ++i) {
    detector_->infer(image);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < benchmark_iterations; ++i) {
    detector_->infer(image);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration<double, std::milli>(end - start);

  double avg_time = total_duration.count() / benchmark_iterations;
  double fps = 1000.0 / avg_time;

  std::cout << "Benchmark Results:" << std::endl;
  std::cout << "  Iterations: " << benchmark_iterations << std::endl;
  std::cout << "  Total time: " << total_duration.count() << " ms" << std::endl;
  std::cout << "  Average time per inference: " << avg_time << " ms" << std::endl;
  std::cout << "  Throughput: " << fps << " FPS" << std::endl;
}

TEST_F(SCNNTrtBackendTest, TestMultipleImages)
{
  std::vector<std::string> test_images = {
    "image_000.png",
    "image_001.png"
  };

  int successful_tests = 0;

  for (const auto & image_path : test_images) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cout << "Skipping missing image: " << image_path << std::endl;
      continue;
    }

    try {
      auto result = detector_->infer(image);
      auto overlay = scnn_trt_backend::utils::create_overlay(image, result.seg_pred);

      // Print existence probabilities for each image
      std::cout << "Image: " << image_path << " - ";
      print_exist_pred(result.exist_pred);

      // Save results with image-specific suffix
      std::string suffix = "_" + std::to_string(successful_tests);
      save_results(image, result.seg_pred, overlay, suffix);

      successful_tests++;

    } catch (const std::exception & e) {
      FAIL() << "Failed to process image " << image_path << ": " << e.what();
    }
  }

  EXPECT_GT(successful_tests, 0) << "No test images were successfully processed";
  std::cout << "Successfully processed " << successful_tests << " test images" << std::endl;
}
