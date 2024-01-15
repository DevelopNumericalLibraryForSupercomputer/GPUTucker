#ifndef CUDA_HELPER_HPP_
#define CUDA_HELPER_HPP_


#include <cstring>

#include <cuda_runtime_api.h>

namespace common {
namespace cuda {

/* helper */
#define _ERROR_TO_STRING(err) \
case err:                   \
  return #err;
  const char *_cuda_get_error_enum(cudaError_t err);

  void _cuda_check(cudaError_t result, char *const func, const char *const file, int const line);
#define _CUDA_API_CALL(val) _cuda_check((val), #val, __FILE__, __LINE__)

  /* Memory */
  void *device_malloc(size_t size);
  void device_free(void *p) noexcept;
  void *pinned_malloc(size_t size);
  void pinned_free(void *p) noexcept;

  size_t get_available_device_memory();

  /* Copy */
  void h2dcpy(void *dst, const void *src, size_t size);
  void h2dcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream);
  void d2hcpy(void *dst, const void *src, size_t size);
  void d2hcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream);
  void h2dcpy_symbol(void *symbol, const void *src, size_t size);

  /* Memset */
  void device_memset(void *dst, int value, size_t size);

  /* Synchronize */
  void stream_synchronize(cudaStream_t const &stream);
  void device_synchronize() noexcept;

  /* Device properties */
  void device_count(int *out);
  void set_device(int device_id) noexcept;

  void destory_streams(cudaStream_t *streams, size_t count) noexcept;

} // namespace cuda
} // namespace common

#endif /* CUDA_HELPER_HPP_ */