#include <cuda_runtime_api.h>

#include <iostream>

#include "common/human_readable.hpp"
#include "gputucker/cmdline_opts.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/optimizer.hpp"
#include "gputucker/tensor.hpp"
#include "gputucker/tensor_manager.hpp"
#include "gputucker/tucker.cuh"
#include "gputucker/cuda_agent.hpp"
#include "gputucker/scheduler.hpp"
#include  "common/cuda_helper.hpp"

int main(int argc, char* argv[]) {
  using namespace supertensor::gputucker;

  CommandLineOptions* options = new CommandLineOptions;
  CommandLineOptions::ReturnStatus ret = options->Parse(argc, argv);
  // TODO rank and gpu options are not being parsed correctly
  if (CommandLineOptions::OPTS_SUCCESS == ret) {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = uint32_t;
    using value_t = double;
    using block_t = Block<index_t, value_t>;
    using tensor_t = Tensor<block_t>;
    using tensor_manager_t = TensorManager<tensor_t>;
    using optimizer_t = Optimizer<tensor_t>;
    using cuda_agent_t = CudaAgent<tensor_t>;
    using scheduler_t = Scheduler<tensor_t, optimizer_t>;

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
    // input_tensor->ToString();

   /* Initialize Cuda Agents */
    MYPRINT("\t... Initialize CUDA Agents\n");

    cuda_agent_t** cuda_agents = allocate<cuda_agent_t *>(options->get_gpu_count());
    for (unsigned dev_id = 0; dev_id < options->get_gpu_count(); ++dev_id)
    {
      cuda_agents[dev_id] = new cuda_agent_t(dev_id);
      cuda_agents[dev_id]->AllocateMaximumBuffer();
    }
    
    size_t avail_gpu_mem = cuda_agents[0]->get_allocated_size();
    std::cout << "Available GPU memory: "
            << common::HumanReadable{(uintmax_t)avail_gpu_mem} << std::endl;

    // Find optimal partition parameters from optimizer
    optimizer_t* optimizer = new optimizer_t;
    optimizer->Initialize(options->get_gpu_count(), options->get_rank(), avail_gpu_mem, input_tensor);
    index_t* partition_dims = optimizer->FindPartitionParms();

    // Create tensor blocks ( = sub-tensors )
    tensor_t* tensor_blocks = new tensor_t(input_tensor);
    tensor_manager->CreateTensorBlocks<optimizer_t>(&input_tensor, &tensor_blocks, optimizer);
    tensor_blocks->ToString();

  // TODO block scheduling
    MYPRINT("\t... Initialize Scheduler\n");
    scheduler_t* scheduler = new scheduler_t;
    scheduler->Initialize(options->get_gpu_count());
    scheduler->Schedule(tensor_blocks, optimizer);

 
    TuckerDecomposition<tensor_t, optimizer_t, cuda_agent_t, scheduler_t>(tensor_blocks,
                                                                          optimizer,
                                                                          cuda_agents,
                                                                          scheduler);

  } else {
    std::cout << "ERROR - problem with options." << std::endl;
  }

  return 0;
}
