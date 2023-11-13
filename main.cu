#include <iostream>

#include "gputucker/cmdline_opts.hpp"
#include "gputucker/tensor.hpp"

int main(int argc, char* argv[]) {
  using namespace supertensor;

  gputucker::CommandLineOptions* options = new gputucker::CommandLineOptions;
  gputucker::CommandLineOptions::ReturnStatus ret = options->parse(argc, argv);
  if (gputucker::CommandLineOptions::OPTS_SUCCESS == ret) {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = size_t;
    using value_t = double;
    using block_t = gputucker::Block<index_t, value_t>;
    using tensor_t = gputucker::Tensor<block_t>;

    bool is_double = std::is_same<value_t, double>::value;
    if (is_double) {
      printf("Values are double type.\n");
    } else {
      printf("Values are float type.\n");
    }

  } else {
    std::cout << "ERROR - problem with options." << std::endl;
  }
  return 0;
}
