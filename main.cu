#include <cuda_runtime_api.h>

#include <iostream>

#include "common/human_readable.hpp"
#include "gputucker/cmdline_opts.hpp"
#include "gputucker/helper.hpp"
#include "gputucker/optimizer.hpp"
#include "gputucker/tensor.hpp"
#include "gputucker/tensor_reader.hpp"

int main(int argc, char* argv[]) {
  using namespace supertensor;

  gputucker::CommandLineOptions* options = new gputucker::CommandLineOptions;
  gputucker::CommandLineOptions::ReturnStatus ret = options->parse(argc, argv);
  // TODO rank and gpu options are not being parsed correctly
  if (gputucker::CommandLineOptions::OPTS_SUCCESS == ret) {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = size_t;
    using value_t = double;
    using block_t = gputucker::Block<index_t, value_t>;
    using tensor_t = gputucker::Tensor<block_t>;
    using tensor_reader_t = gputucker::TensorReader<tensor_t>;
    using optimizer_t = gputucker::Optimizer<tensor_t>;

    bool is_double = std::is_same<value_t, double>::value;
    if (is_double) {
      printf("Values are double type.\n");
    } else {
      printf("Values are float type.\n");
    }

    // Read tensor from file
    tensor_reader_t* reader = new tensor_reader_t(options->get_order());
    reader->parse_from_file(options->get_input_path());
    reader->to_string();

    // Allocate memory on GPU
    void* p;
    size_t avail_gpu_mem, total_gpu_mem;
    cudaSetDevice(0);
    gputucker::CUDA_API_CALL(cudaSetDevice(0));
    gputucker::CUDA_API_CALL(cudaMemGetInfo(&avail_gpu_mem, &total_gpu_mem));
    avail_gpu_mem = 0.9 * avail_gpu_mem;
    gputucker::CUDA_API_CALL(cudaMalloc(&p, avail_gpu_mem));

    std::cout << "Available GPU memory: "
              << common::HumanReadable{(uintmax_t)avail_gpu_mem} << std::endl;

    // Create optimizer
    optimizer_t* optimizer = new optimizer_t;
    optimizer->initialize(reader->data, options->get_gpu_count(), 50,
                          avail_gpu_mem);
    optimizer->find_partition_parms();

  } else {
    std::cout << "ERROR - problem with options." << std::endl;
  }
  return 0;
}
