#include <cuda_runtime_api.h>

#include <iostream>

#include "common/human_readable.hpp"
#include "gputucker/cmdline_opts.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/optimizer.hpp"
#include "gputucker/tensor.hpp"
#include "gputucker/tensor_manager.hpp"

int main(int argc, char* argv[]) {
  using namespace supertensor::gputucker;

  CommandLineOptions* options = new CommandLineOptions;
  CommandLineOptions::ReturnStatus ret = options->Parse(argc, argv);
  // TODO rank and gpu options are not being parsed correctly
  if (CommandLineOptions::OPTS_SUCCESS == ret) {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = size_t;
    using value_t = double;
    using block_t = Block<index_t, value_t>;
    using tensor_t = Tensor<block_t>;
    using tensor_manager_t = TensorManager<tensor_t>;
    using optimizer_t = Optimizer<tensor_t>;

    bool is_double = std::is_same<value_t, double>::value;
    if (is_double) {
      printf("Values are double type.\n");
    } else {
      printf("Values are float type.\n");
    }

    // Read tensor from file
    tensor_manager_t* tensor_manager = new tensor_manager_t;
    tensor_t* input_tensor = new tensor_t(options->get_order());
    tensor_manager->ParseFromFile(options->get_input_path(), &input_tensor);
    input_tensor->ToString();

    // Allocate memory on GPU
    void* p;
    size_t avail_gpu_mem, total_gpu_mem;
    cudaSetDevice(0);
    CUDA_API_CALL(cudaSetDevice(0));
    CUDA_API_CALL(cudaMemGetInfo(&avail_gpu_mem, &total_gpu_mem));
    avail_gpu_mem = 0.9 * avail_gpu_mem;
    CUDA_API_CALL(cudaMalloc(&p, avail_gpu_mem));

    std::cout << "Available GPU memory: "
              << common::HumanReadable{(uintmax_t)avail_gpu_mem} << std::endl;

    // Create optimizer
    optimizer_t* optimizer = new optimizer_t;
    optimizer->Initialize(options->get_gpu_count(), options->get_rank(), avail_gpu_mem);
    optimizer->FindPartitionParms(input_tensor);

    tensor_t* tensor_blocks = new tensor_t(options->get_order());
    tensor_manager->CreateTensorBlocks(&input_tensor, &tensor_blocks, optimizer->partition_dims);

  } else {
    std::cout << "ERROR - problem with options." << std::endl;
  }
  return 0;
}
