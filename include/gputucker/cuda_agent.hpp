#ifndef CUDA_AGENT_HPP_
#define CUDA_AGENT_HPP_

#include "cuda_runtime_api.h"

#include "common/memory_region.hpp"

namespace supertensor {
namespace gputucker {

  class CudaAgent {
  public:
    CudaAgent();
    ~CudaAgent();

    bool AllocateMaximumBuffer();
    bool CreateCudaStream();
    void ToString();

    int get_device_id() const;
    cudaStream_t *get_cuda_streams();
    cudaDeviceProp *get_device_properties();
    memrgn_t *get_base_memory_region();

  private:
    int _device_id;
    cudaStream_t *_cuda_streams;
    cudaDeviceProp *_device_properties;
    memrgn_t *_base_mr;

  }; // class CudaAgent

}
}