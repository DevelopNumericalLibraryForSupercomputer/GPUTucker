#ifndef OPTIMIZER_HPP_
#define OPTIMIZER_HPP_

#include "common/human_readable.hpp"
#include "gputucker/constants.hpp"
#include "gputucker/tensor.hpp"

namespace supertensor {
namespace gputucker {

#define OPTIMIZER_TEMPLATE template <typename TensorType>
#define OPTIMIZER_TEMPLATE_ARGS TensorType

OPTIMIZER_TEMPLATE
class Optimizer {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

 public:
  enum Component { SubTensor, CoreTensor, SubFactors, Delta, ComponentCount };
  std::string str_Component[Component::ComponentCount] = { "SubTensor", "CoreTensor", "SubFactors", "Delta"};

  struct CostMetric {
    size_t data_size;
    size_t transfer_size;

    CostMetric(size_t new_data_size, size_t new_transfer_size)
        : data_size(new_data_size), transfer_size(new_transfer_size) {}
    CostMetric() : CostMetric(0, 0) {}
    void to_string() {
      std::cout << "-> Data size: "
                << common::HumanReadable{(std::uintmax_t)this->data_size} << std::endl;
      std::cout << "-> Transfer size: "
                << common::HumanReadable{(std::uintmax_t)this->transfer_size} << std::endl;
    }
  };  // struct CostMetric

 public:
  Optimizer() {}
  ~Optimizer() {}
  void Initialize(tensor_t *tensor, unsigned short new_gpu_count,
                  unsigned int new_rank, uint64_t new_gpu_mem_size);
  void Initialize(unsigned short new_order, index_t *new_dims,
                  uint64_t new_nnz_count, unsigned short new_gpu_count,
                  unsigned int new_rank, uint64_t new_gpu_mem_size);

  size_t get_data_size(Component comp);
  size_t get_transfer_size(Component comp);

  size_t get_all_data_size();
  size_t get_all_transfer_size();

  void estimate_component_costs();
  void find_partition_parms();
  void set_available_nnz_count_per_task();

  void determine_partition_type();  // which is the partition type for the input
                                    // tensor?
  void to_string();

 private:
  /* Get data size for a CUDA execution sequence*/
  size_t _get_data_size_input_tensor();
  size_t _get_data_size_sub_tensor();
  size_t _get_data_size_core_tensor();
  size_t _get_data_size_factors();
  size_t _get_data_size_each_sub_factor();
  size_t _get_data_size_delta();
  size_t _get_data_size_sub_delta();  // intermediate data size using
                                      // this->avail_nnz_count;

  size_t _get_transfer_size_sub_tensor();
  size_t _get_transfer_size_core_tensor();
  size_t _get_transfer_size_sub_factors();
  size_t _get_transfer_size_delta();  // intermediate data size

  /* determining the next axis or dimension along which the data will be partitioned. */
  unsigned short _get_next_partition_axis();

  /* Adjusting block dimensions using partition dimensions */
  void _update_block_dims();

  void _determine_partition_type();

 public:
  unsigned int rank;
  unsigned short gpu_count;
  unsigned short order;
  index_t *dims;
  uint64_t nnz_count;

  index_t *block_dims;
  index_t *partition_dims;
  uint64_t block_count;

  uint64_t predicted_nnz_count;  // (estimated) nonzeros per a task
  int cuda_stream_count;
  uint64_t avail_nnz_count_per_task;

  CostMetric *component_cost;
  gputucker::enums::PartitionTypes partition_type;

 private:
  size_t _gpu_mem_size;
};
}  // namespace gputucker
}  // namespace supertensor

#include "gputucker/optimizer.tpp"
#endif /* OPTIMIZER_HPP_ */