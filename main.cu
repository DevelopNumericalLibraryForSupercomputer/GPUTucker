#include <cuda_runtime_api.h>

#include <iostream>

#include "gputucker/cmdline_opts.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/tensor.hpp"
#include "gputucker/tensor_manager.hpp"
#include "gputucker/tucker.cuh"

int main(int argc, char *argv[]) {
  using namespace supertensor::gputucker;

  // Parse command line options
  CommandLineOptions *options = new CommandLineOptions;
  CommandLineOptions::ReturnStatus ret = options->Parse(argc, argv);

  if (CommandLineOptions::OPTS_SUCCESS == ret) {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = uint32_t;
    using value_t = double;
    using block_t = Block<index_t, value_t>;
    using tensor_t = Tensor<block_t>;

    bool is_double = std::is_same<value_t, double>::value;
    if (is_double) {
      printf("Values are double type.\n");
    } else {
      printf("Values are float type.\n");
    }

    // Read tensor from file
    tensor_t *input_tensor = new tensor_t(options->get_order());
    ParseFromFile<tensor_t>(options->get_input_path(), &input_tensor);

    // Perform Tucker decomposition
    TuckerDecomposition<tensor_t>(input_tensor, options->get_rank(), options->get_gpu_count());

  } else {
    std::cout << "ERROR - problem with options." << std::endl;
  }

  return 0;
}
