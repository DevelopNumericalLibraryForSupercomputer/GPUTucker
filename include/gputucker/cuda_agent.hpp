#ifndef CUDA_AGENT_HPP_
#define CUDA_AGENT_HPP_

#include "cuda_runtime_api.h"

#include "common/memory_region.hpp"

namespace supertensor {
namespace gputucker {

  class CudaAgent {
  public:
    using memrgn_t = common::MemoryRegion<char>;
    
    CudaAgent();
    CudaAgent(unsigned id);
    ~CudaAgent();

    bool AllocateMaximumBuffer();
    bool CreateCudaStream(unsigned int stream_count);
    void ToString();

    int get_device_id() const { return this->_device_id; }
    cudaStream_t *get_cuda_streams() { return this->_streams; }
    cudaDeviceProp *get_device_properties() { return &this ->_device_properties; }
    memrgn_t *get_base_memory_region() { return this->_base_mr; }

  private:
    int _device_id;
    unsigned _stream_count;
    cudaStream_t *_streams;
    cudaDeviceProp _device_properties;
    memrgn_t *_base_mr;

  }; // class CudaAgent

}
}


#endif // CUDA_AGENT_HPP_