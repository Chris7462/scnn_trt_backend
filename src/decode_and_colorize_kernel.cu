#include "scnn_trt_backend/config.hpp"
#include "scnn_trt_backend/decode_and_colorize_kernel.hpp"


namespace scnn_trt_backend
{

// Declare constant memory for lane colormap (5 classes)
__constant__ uchar3 d_colormap[5];

// Initialize constant memory (call once during initialization)
void initialize_colormap_constants()
{
  // Allocate and initialize GPU colormap (one-time initialization)
  // Note: config uses RGB format, we convert to BGR for OpenCV output
  uchar3 h_colormap[5];
  for (int i = 0; i < 5; ++i) {
    h_colormap[i] = {config::LANE_COLORMAP[i][2],   // B
                     config::LANE_COLORMAP[i][1],   // G
                     config::LANE_COLORMAP[i][0]};  // R
  }
  cudaMemcpyToSymbol(d_colormap, h_colormap, 5 * sizeof(uchar3));
}

// Device function: sigmoid activation
__device__ __forceinline__ float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

// CUDA kernel for lane segmentation decode and colorize with existence filtering
__global__ void decode_and_colorize_kernel(
  const float * seg_input,
  const float * exist_input,
  uchar3 * output,
  int width, int height,
  int num_classes, int num_lanes,
  float exist_threshold)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * width + x;

  if (x >= width || y >= height) {
    return;
  }

  // Argmax over classes
  int best_class = 0;
  float max_score = seg_input[idx];
  for (int c = 1; c < num_classes; ++c) {
    float score = seg_input[c * width * height + idx];
    if (score > max_score) {
      max_score = score;
      best_class = c;
    }
  }

  // Check lane existence (best_class 1-4 corresponds to lanes 0-3)
  // If best_class is 0 (background), always use background color
  // If best_class is 1-4, check if that lane exists
  if (best_class > 0 && best_class <= num_lanes) {
    int lane_idx = best_class - 1;  // Convert class (1-4) to lane index (0-3)
    float exist_prob = sigmoid(exist_input[lane_idx]);

    // If lane doesn't exist, treat as background
    if (exist_prob <= exist_threshold) {
      best_class = 0;
    }
  }

  output[idx] = d_colormap[best_class];
}

void launch_decode_and_colorize_kernel(
  const float * seg_input_gpu,
  const float * exist_input_gpu,
  uchar3 * output_gpu,
  int width, int height,
  int num_classes, int num_lanes,
  float exist_threshold,
  cudaStream_t stream)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y);

  decode_and_colorize_kernel<<<gridSize, blockSize, 0, stream>>>(
    seg_input_gpu, exist_input_gpu, output_gpu,
    width, height,
    num_classes, num_lanes,
    exist_threshold);
}

}  // namespace scnn_trt_backend
