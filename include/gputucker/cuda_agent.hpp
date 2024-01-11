#ifndef CUDA_AGENT_HPP_
#define CUDA_AGENT_HPP_

#include "cuda_runtime_api.h"

#include "common/memory_region.hpp"
#include "gputucker/tensor.hpp"

namespace supertensor {
namespace gputucker {

  #define CUDAAGENT_TEMPLATE \
    template <typename TensorType>
  #define CUDAAGENT_TEMPLATE_ARGS \
    TensorType

  CUDAAGENT_TEMPLATE
  class CudaAgent {
  public:
    using tensor_t = TensorType;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;
    using memrgn_t = common::MemoryRegion<char>;

    struct DeviceBuffer {

      using int_mr_t = common::MemoryRegion<int>;
      using index_mr_t = common::MemoryRegion<index_t>;
      using value_mr_t = common::MemoryRegion<value_t>;
      using addr_mr_t = common::MemoryRegion<std::uintptr_t *>;

      index_mr_t X_indices[gputucker::constants::kMaxOrder];
      value_mr_t X_values;
      addr_mr_t X_idx_addr;

      index_mr_t core_indices[gputucker::constants::kMaxOrder];
      value_mr_t core_values;
      addr_mr_t core_idx_addr;

      value_mr_t factors[gputucker::constants::kMaxOrder];
      addr_mr_t factor_addr;

      value_mr_t delta;
    };
    
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

#include "gputucker/cuda_agent.tpp"

#endif // CUDA_AGENT_HPP_