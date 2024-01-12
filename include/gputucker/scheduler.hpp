#ifndef SCHEDULER_HPP_
#define SCHEDULER_HPP_

#include <vector>

#include "gputucker/tensor.hpp"

namespace supertensor {
namespace gputucker {

#define SCHEDULER_TEMPLATE \
  template <typename TensorType, typename OptimizerType>
#define SCHEDULER_TEMPLATE_ARGS \
  TensorType, OptimizerType

  SCHEDULER_TEMPLATE
  class Scheduler  {
    using tensor_t = TensorType;
    using index_t = typename tensor_t::index_t;
    using value_t = typename tensor_t::value_t;
    using optimizer_t = OptimizerType;

  public:
    Scheduler() {
    }
    ~Scheduler() {
    }
    void Initialize(unsigned short new_gpu_count);
    void Schedule(tensor_t *tensor, optimizer_t *optimizer);

    private:
    void _NonzeroBasedPartitioning(tensor_t *tensor, optimizer_t *optimizer);
    void _DimensionBasedPartitioning(tensor_t *tensor, optimizer_t *optimizer);

  public:
    struct Task {
      uint64_t block_id;
      uint64_t nnz_count;
      uint64_t offset; // default: 0, if dense tensor is from 0 to iter - 1.
      int stream_offset;
      Task(uint64_t new_block_id, uint64_t new_nnz_count, uint64_t new_offset, int new_stream_offset)
          : block_id(new_block_id), nnz_count(new_nnz_count), offset(new_offset), stream_offset(new_stream_offset) {
      }
      void ToString() {
        printf("[%lu]-block \t %lu nnzs \t %lu offset \t %d streams\n", block_id, nnz_count, offset, stream_offset);
      }
    };

  public:
    std::vector<Task> *tasks; // number of gpus
    uint64_t task_count;      // >= block_count
    unsigned short gpu_count;
    uint64_t nnz_count_per_task; 
  };
  
}
}

#include "gputucker/scheduler.tpp"

#endif /* SCHEDULER_HPP_ */