#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime_api.h>

#include "common/cuda_helper.h"
#include "common/human_readable.h"
#include "common/size.h"

namespace common {
namespace cuda {

  const char *_cuda_get_error_enum(cudaError_t err) {
    switch (err) {
      _ERROR_TO_STRING(cudaSuccess)
      _ERROR_TO_STRING(cudaErrorMissingConfiguration)
      _ERROR_TO_STRING(cudaErrorMemoryAllocation)
      _ERROR_TO_STRING(cudaErrorInitializationError)
      _ERROR_TO_STRING(cudaErrorLaunchFailure)
      _ERROR_TO_STRING(cudaErrorPriorLaunchFailure)
      _ERROR_TO_STRING(cudaErrorLaunchTimeout)
      _ERROR_TO_STRING(cudaErrorLaunchOutOfResources)
      _ERROR_TO_STRING(cudaErrorInvalidDeviceFunction)
      _ERROR_TO_STRING(cudaErrorInvalidConfiguration)
      _ERROR_TO_STRING(cudaErrorInvalidDevice)
      _ERROR_TO_STRING(cudaErrorInvalidValue)
      _ERROR_TO_STRING(cudaErrorInvalidPitchValue)
      _ERROR_TO_STRING(cudaErrorInvalidSymbol)
      _ERROR_TO_STRING(cudaErrorMapBufferObjectFailed)
      _ERROR_TO_STRING(cudaErrorUnmapBufferObjectFailed)
      _ERROR_TO_STRING(cudaErrorInvalidHostPointer)
      _ERROR_TO_STRING(cudaErrorInvalidDevicePointer)
      _ERROR_TO_STRING(cudaErrorInvalidTexture)
      _ERROR_TO_STRING(cudaErrorInvalidTextureBinding)
      _ERROR_TO_STRING(cudaErrorInvalidChannelDescriptor)
      _ERROR_TO_STRING(cudaErrorInvalidMemcpyDirection)
      _ERROR_TO_STRING(cudaErrorAddressOfConstant)
      _ERROR_TO_STRING(cudaErrorTextureFetchFailed)
      _ERROR_TO_STRING(cudaErrorTextureNotBound)
      _ERROR_TO_STRING(cudaErrorSynchronizationError)
      _ERROR_TO_STRING(cudaErrorInvalidFilterSetting)
      _ERROR_TO_STRING(cudaErrorInvalidNormSetting)
      _ERROR_TO_STRING(cudaErrorMixedDeviceExecution)
      _ERROR_TO_STRING(cudaErrorCudartUnloading)
      _ERROR_TO_STRING(cudaErrorUnknown)
      _ERROR_TO_STRING(cudaErrorNotYetImplemented)
      _ERROR_TO_STRING(cudaErrorMemoryValueTooLarge)
      _ERROR_TO_STRING(cudaErrorInvalidResourceHandle)
      _ERROR_TO_STRING(cudaErrorNotReady)
      _ERROR_TO_STRING(cudaErrorInsufficientDriver)
      _ERROR_TO_STRING(cudaErrorSetOnActiveProcess)
      _ERROR_TO_STRING(cudaErrorInvalidSurface)
      _ERROR_TO_STRING(cudaErrorNoDevice)
      _ERROR_TO_STRING(cudaErrorECCUncorrectable)
      _ERROR_TO_STRING(cudaErrorSharedObjectSymbolNotFound)
      _ERROR_TO_STRING(cudaErrorSharedObjectInitFailed)
      _ERROR_TO_STRING(cudaErrorUnsupportedLimit)
      _ERROR_TO_STRING(cudaErrorDuplicateVariableName)
      _ERROR_TO_STRING(cudaErrorDuplicateTextureName)
      _ERROR_TO_STRING(cudaErrorDuplicateSurfaceName)
      _ERROR_TO_STRING(cudaErrorDevicesUnavailable)
      _ERROR_TO_STRING(cudaErrorInvalidKernelImage)
      _ERROR_TO_STRING(cudaErrorNoKernelImageForDevice)
      _ERROR_TO_STRING(cudaErrorIncompatibleDriverContext)
      _ERROR_TO_STRING(cudaErrorPeerAccessAlreadyEnabled)
      _ERROR_TO_STRING(cudaErrorPeerAccessNotEnabled)
      _ERROR_TO_STRING(cudaErrorDeviceAlreadyInUse)
      _ERROR_TO_STRING(cudaErrorProfilerDisabled)
      _ERROR_TO_STRING(cudaErrorProfilerNotInitialized)
      _ERROR_TO_STRING(cudaErrorProfilerAlreadyStarted)
      _ERROR_TO_STRING(cudaErrorProfilerAlreadyStopped)
      _ERROR_TO_STRING(cudaErrorAssert)
      _ERROR_TO_STRING(cudaErrorTooManyPeers)
      _ERROR_TO_STRING(cudaErrorHostMemoryAlreadyRegistered)
      _ERROR_TO_STRING(cudaErrorHostMemoryNotRegistered)
      _ERROR_TO_STRING(cudaErrorStartupFailure)
      _ERROR_TO_STRING(cudaErrorApiFailureBase)
    }
    return "<unknown>";
  }

  void _cuda_check(cudaError_t result, char *const func, const char *const file, int const line) {
    if (result) {
      std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                    file, line, static_cast<unsigned int>(result), _cuda_get_error_enum(result), func);
      cudaDeviceReset();
      std::exit(EXIT_FAILURE);
    }
  }

  /* Memory */
  void *device_malloc(size_t size) {
    void *p;
    _CUDA_API_CALL(cudaMalloc(&p, size));
    return p;
  }

  void device_free(void *p) noexcept {
    if (p == nullptr) {
      throw std::runtime_error(std::string("Memory Deallocation ERROR \n\t [ptr != NULL]"));
    }

    _CUDA_API_CALL(cudaFree(p));
  }

  void *pinned_malloc(size_t size) {
    void *p;
    _CUDA_API_CALL(cudaHostAlloc(&p, size, cudaHostAllocPortable));
    return p;
  }

  void pinned_free(void *p) noexcept {
    if (p == nullptr) {
      throw std::runtime_error(std::string("Memory Deallocation ERROR \n\t [ptr != NULL]"));
    }
    _CUDA_API_CALL(cudaFreeHost(p));
  }

  size_t get_available_device_memory() {
    size_t avail, total, used;
    _CUDA_API_CALL(cudaMemGetInfo(&avail, &total));
    used = total - avail;
    std::cout << "Device memory:\n\t used " << common::HumanReadable{used}
              << "\n\t available " << common::HumanReadable{avail}
              << "\n\t total " << common::HumanReadable{total} << std::endl;
    return avail - common::MiB(128);
  }

  /* Copy */
  void h2dcpy(void *dst, const void *src, size_t size) {
    _CUDA_API_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  }

  void h2dcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream) {
    _CUDA_API_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
  }

  void d2hcpy(void *dst, const void *src, size_t size) {
    _CUDA_API_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }

  void d2hcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream) {
    _CUDA_API_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
  }

  void h2dcpy_symbol(void *symbol, const void *src, size_t size) {
    _CUDA_API_CALL(cudaMemcpyToSymbol(symbol, src, size, 0, cudaMemcpyHostToDevice));
  }

  /* Memset */
  void device_memset(void *dst, int value, size_t size) {
    _CUDA_API_CALL(cudaMemset(dst, value, size));
  }
  // void device_memset_async() {
  // }

  /* Synchronize */
  void stream_synchronize(cudaStream_t const &stream) {
    _CUDA_API_CALL(cudaStreamSynchronize(stream));
  }

  void device_synchronize() noexcept {
    _CUDA_API_CALL(cudaDeviceSynchronize());
  }

  /* Device properties */
  void device_count(int *out) {
    _CUDA_API_CALL(cudaGetDeviceCount(out));
    if (out == 0) {
      printf("There are no available device(s) that support CUDA\n");
    }
    else {
      printf("Detected %d CUDA Capable device(s)\n", *out);
    }
  }

  void set_device(int device_id) noexcept {
    _CUDA_API_CALL(cudaSetDevice(device_id));
  }

  void destory_streams(cudaStream_t *streams, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
      _CUDA_API_CALL(cudaStreamDestroy(streams[i]));
    }
  }

} // namespace cuda
} // namespace common
