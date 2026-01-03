#pragma once

#include <cuda_runtime.h>


namespace scnn_trt_backend
{

void initialize_colormap_constants();

/**
 * @brief GPU accelerated lane segmentation decode and colorize kernel
 * @param seg_input_gpu   Segmentation logits on GPU: shape [num_classes, height, width]
 * @param exist_input_gpu Existence logits on GPU: shape [num_lanes]
 * @param output_gpu      Output buffer on GPU: shape [height * width], CV_8UC3
 * @param width           Image width
 * @param height          Image height
 * @param num_classes     Number of segmentation classes (5: background + 4 lanes)
 * @param num_lanes       Number of lanes (4)
 * @param exist_threshold Threshold for lane existence (default: 0.5)
 * @param stream          CUDA stream to launch the kernel on
 */
void launch_decode_and_colorize_kernel(
  const float * seg_input_gpu,
  const float * exist_input_gpu,
  uchar3 * output_gpu,
  int width, int height,
  int num_classes, int num_lanes,
  float exist_threshold,
  cudaStream_t stream);

}  // namespace scnn_trt_backend
