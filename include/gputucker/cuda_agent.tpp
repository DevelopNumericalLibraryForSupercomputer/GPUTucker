#include <iostream>
#include <cassert>
#include <cuda_runtime_api.h>

#include "gputucker/helper.hpp"
#include "gputucker/cuda_agent.hpp"
#include "common/cuda_helper.hpp"
#include "common/memory_region.hpp"

namespace supertensor {
namespace gputucker {

CUDAAGENT_TEMPLATE
CudaAgent<CUDAAGENT_TEMPLATE_ARGS>::CudaAgent(unsigned id) : _device_id(id){
  CUDA_API_CALL(cudaSetDevice(this->_device_id));
  CUDA_API_CALL(cudaGetDeviceProperties(&this->_device_properties, this->_device_id));
}

CUDAAGENT_TEMPLATE
CudaAgent<CUDAAGENT_TEMPLATE_ARGS>::CudaAgent() : CudaAgent(0){
}

CUDAAGENT_TEMPLATE
bool CudaAgent<CUDAAGENT_TEMPLATE_ARGS>::AllocateMaximumBuffer() {
  size_t avail, total, used;
  CUDA_API_CALL(cudaSetDevice(this->_device_id));
  avail = common::cuda::get_available_device_memory();

  this->_base_mr = new memrgn_t(avail, 1);
  this->_allocated_size = this->_base_mr->get_size(); // aligned available memory
  this->_base_mr->set_ptr(common::cuda::device_malloc(this->_allocated_size));

  return false;
}

CUDAAGENT_TEMPLATE
bool CudaAgent<CUDAAGENT_TEMPLATE_ARGS>::CreateCudaStream(unsigned int stream_count) {
  assert(stream_count > 0);
  this->_stream_count = stream_count;
  this->_streams = static_cast<cudaStream_t *>(malloc(this->_stream_count * sizeof(cudaStream_t)));
  if (this->_streams == nullptr) {
    throw std::runtime_error(ERROR_LOG("[ERROR]Failed to allocate buffer for CUDA streams!"));
    return false;
  }

  for (size_t i = 0; i < this->_stream_count; ++i) {
    CUDA_API_CALL(cudaStreamCreateWithFlags(&this->_streams[i], cudaStreamNonBlocking));
  }
  return true;
}

CUDAAGENT_TEMPLATE
void CudaAgent<CUDAAGENT_TEMPLATE_ARGS>::SetDeviceBuffers(tensor_t* tensor, int rank, uint64_t max_nnz_count_in_block) {

  int order = tensor->order;
  index_t *block_dims = tensor->block_dims;
  uint64_t core_size = std::pow(rank, order);

  CUDA_API_CALL(cudaSetDevice(this->_device_id));
  CreateCudaStream(this->_stream_count);

  memrgn_t *tmp_base_mr = new memrgn_t();
  tmp_base_mr->set_memory_region(this->_base_mr);
  
  // For addr. memory regions
  dev_buf.X_idx_addr.Initialize(tmp_base_mr->get_shift_ptr(),
                                  sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder,
                                  this->_stream_count);
  tmp_base_mr->shift_ptr(dev_buf.X_idx_addr.get_total_size());

  dev_buf.core_idx_addr.Initialize(tmp_base_mr->get_shift_ptr(),
                                  sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder,
                                  1);
  tmp_base_mr->shift_ptr(dev_buf.core_idx_addr.get_total_size());

  dev_buf.factor_addr.Initialize(tmp_base_mr->get_shift_ptr(),
                                sizeof(std::uintptr_t *) * gputucker::constants::kMaxOrder,
                                this->_stream_count);
  tmp_base_mr->shift_ptr(dev_buf.factor_addr.get_total_size());

  // For X indices
  for (int axis = 0; axis < order; ++axis) {
    dev_buf.X_indices[axis].Initialize(tmp_base_mr->get_shift_ptr(),
                                        sizeof(index_t) * max_nnz_count_in_block,
                                        this->_stream_count);
    tmp_base_mr->shift_ptr(dev_buf.X_indices[axis].get_total_size());
  }

  // For X values
  dev_buf.X_values.Initialize(tmp_base_mr->get_shift_ptr(),
                              sizeof(value_t) * max_nnz_count_in_block,
                              this->_stream_count);
  tmp_base_mr->shift_ptr(dev_buf.X_values.get_total_size());

  // For core tensor
  for (int axis = 0; axis < order; ++axis) {
    dev_buf.core_indices[axis].Initialize(tmp_base_mr->get_shift_ptr(),
                                          sizeof(index_t) * core_size,
                                          1);
    tmp_base_mr->shift_ptr(dev_buf.core_indices[axis].get_total_size());
  }
  // For core values
  dev_buf.core_values.Initialize(tmp_base_mr->get_shift_ptr(),
                                sizeof(value_t) * core_size,
                                1);
  tmp_base_mr->shift_ptr(dev_buf.core_values.get_total_size());

  // For delta
  dev_buf.delta.Initialize(tmp_base_mr->get_shift_ptr(),
                          sizeof(value_t) * max_nnz_count_in_block * rank,
                          this->_stream_count);

  tmp_base_mr->shift_ptr(dev_buf.delta.get_total_size());

  // For factore matrices
  for (int axis = 0; axis < order; ++axis)
  {
    dev_buf.factors[axis].Initialize(tmp_base_mr->get_shift_ptr(),
                                    sizeof(value_t) * block_dims[axis] * rank,
                                    this->_stream_count);
    tmp_base_mr->shift_ptr(dev_buf.factors[axis].get_total_size());
  }

  auto diff = ((char *)tmp_base_mr->get_shift_ptr() - (char *)this->_base_mr->get_ptr(0));
  std::cout << "\t... Used size in GPU[" << this->_device_id << "]: " << common::HumanReadable{(std::uintmax_t)diff} << std::endl;

}

CUDAAGENT_TEMPLATE
void CudaAgent<CUDAAGENT_TEMPLATE_ARGS>::ToString() {
  auto prop = this->_device_properties;
  printf("\n*-*-*-*-*- General Information for device [%d] *-*-*-*-*-\n", this->_device_id);
  printf("Name:  %s\n", prop.name);
  printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
  printf("Clock rate:  %d\n", prop.clockRate);
  printf("Device copy overlap:  ");
  if (prop.deviceOverlap) {
    printf("Enabled\n");
  }
  else {
    printf("Disabled\n");
  }
  printf("Kernel execution timeout :  ");
  if (prop.kernelExecTimeoutEnabled) {
    printf("Enabled\n");
  }
  else {
    printf("Disabled\n");
  }
  printf("\n");

  printf("*-*-*-*-*- Memory Information for device [%d] *-*-*-*-*-\n", this->_device_id);
  printf("Total global mem:  %ld\n", prop.totalGlobalMem);
  printf("Total constant Mem:  %ld\n", prop.totalConstMem);
  printf("Max mem pitch:  %ld\n", prop.memPitch);
  printf("Texture Alignment:  %ld\n", prop.textureAlignment);
  printf("\n");

  printf("*-*-*-*-*- MP Information for device [%d] *-*-*-*-*-\n", this->_device_id);
  printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
  printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
  printf("Registers per mp:  %d\n", prop.regsPerBlock);
  printf("Threads in warp:  %d\n", prop.warpSize);
  printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
  printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("\n");
}

} // namespace gputucker
} // namespace supertensor