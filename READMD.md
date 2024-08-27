@mainpage

## Using GPUTucker for Tucker Decomposition
This example demonstrates how to use the GPUTucker library for performing Tucker decomposition on large-scale tensors using GPU acceleration. The code showcases the process from setting up command-line options to performing the decomposition and optimizing memory usage on GPUs.

### Code
```cpp
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
#include "common/cuda_helper.hpp"

// Main function
int main(int argc, char* argv[]) {
  using namespace supertensor::gputucker;

  // Parse command-line options
  CommandLineOptions* options = new CommandLineOptions;
  CommandLineOptions::ReturnStatus ret = options->Parse(argc, argv);

  // Check if options were parsed successfully
  if (CommandLineOptions::OPTS_SUCCESS == ret) {
    // Display input file path
    std::cout << options->get_input_path() << std::endl;

    // Define the data types for indices and values, and create necessary types
    using index_t = uint32_t;
    using value_t = double;
    using block_t = Block<index_t, value_t>;
    using tensor_t = Tensor<block_t>;
    using tensor_manager_t = TensorManager<tensor_t>;
    using optimizer_t = Optimizer<tensor_t>;
    using cuda_agent_t = CudaAgent<tensor_t>;
    using scheduler_t = Scheduler<tensor_t, optimizer_t>;

    // Check if the value type is double
    bool is_double = std::is_same<value_t, double>::value;
    if (is_double) {
      printf("Values are double type.\n");
    } else {
      printf("Values are float type.\n");
    }

    // Read the tensor data from the specified file
    tensor_manager_t* tensor_manager = new tensor_manager_t;
    tensor_t* input_tensor = new tensor_t(options->get_order());
    tensor_manager->ParseFromFile(options->get_input_path(), &input_tensor);

    /* Initialize CUDA Agents */
    MYPRINT("\t... Initialize CUDA Agents\n");

    // Allocate memory for CUDA agents and initialize them
    cuda_agent_t** cuda_agents = allocate<cuda_agent_t *>(options->get_gpu_count());
    for (unsigned dev_id = 0; dev_id < options->get_gpu_count(); ++dev_id) {
      cuda_agents[dev_id] = new cuda_agent_t(dev_id);
      cuda_agents[dev_id]->AllocateMaximumBuffer();
    }
    
    // Display available GPU memory
    size_t avail_gpu_mem = cuda_agents[0]->get_allocated_size();
    std::cout << "Available GPU memory: "
              << common::HumanReadable{(uintmax_t)avail_gpu_mem} << std::endl;

    // Initialize the optimizer and find optimal partition parameters
    optimizer_t* optimizer = new optimizer_t;
    optimizer->Initialize(options->get_gpu_count(), options->get_rank(), avail_gpu_mem, input_tensor);
    index_t* partition_dims = optimizer->FindPartitionParms();

    // Create tensor blocks based on the input tensor and partition parameters
    tensor_t* tensor_blocks = new tensor_t(input_tensor);
    tensor_manager->CreateTensorBlocks<optimizer_t>(&input_tensor, &tensor_blocks, optimizer);
    tensor_blocks->ToString();

    /* Initialize the Scheduler */
    MYPRINT("\t... Initialize Scheduler\n");
    scheduler_t* scheduler = new scheduler_t;
    scheduler->Initialize(options->get_gpu_count());
    scheduler->Schedule(tensor_blocks, optimizer);

    // Perform Tucker decomposition using the GPUTucker algorithm
    TuckerDecomposition<tensor_t, optimizer_t, cuda_agent_t, scheduler_t>(
        tensor_blocks,
        optimizer,
        cuda_agents,
        scheduler);

  } else {
    // Display an error message if command-line options were not parsed correctly
    std::cout << "ERROR - problem with options." << std::endl;
  }

  return 0;
}
```


#### Key Components
1. **Command Line Parsing:**
   - The program begins by parsing command-line options, such as the input tensor file, tensor order, Tucker rank, and the number of GPUs to use. These options are critical for configuring the GPUTucker algorithm.

2. **Tensor Management:**
   - The tensor data is read from the specified input file. The tensor manager is responsible for handling the tensor's structure and converting it into a format suitable for processing by the GPUTucker algorithm.

3. **CUDA Agent Initialization:**
   - CUDA agents are initialized to handle GPU resources. Each GPU is assigned an agent, which manages memory allocation and data transfer between the CPU and GPU.

4. **Optimization:**
  - The optimizer determines the best way to partition the tensor across the available GPUs. This step is crucial for balancing the workload and ensuring efficient computation.
  
5. **Scheduling:**
   - A scheduler organizes the computation tasks across GPUs. It ensures that each GPU is assigned tasks efficiently, preventing workload imbalances.

6. **Tucker Decomposition:**
   - The core of the example is the Tucker decomposition, which is performed on the tensor blocks. The decomposition leverages GPU acceleration to achieve high performance.


**If the command-line options are not correctly parsed, the program outputs an error message and exits.**