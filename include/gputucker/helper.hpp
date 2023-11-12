#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>  // std::remove_pointer

namespace supertensor {
namespace gputucker {

// for Colorful print on terminal
#define RED "\x1B[31m"  // red
#define GRN "\x1B[32m"  // green
#define YEL "\x1B[33m"  // yellow
#define BLU "\x1B[34m"  // blue
#define MAG "\x1B[35m"  // magenta
#define CYN "\x1B[36m"  // cyan
#define WHT "\x1B[37m"  // white
#define RESET "\x1B[0m"

#define MYDEBUG(Fmt, ...) \
  { printf(BLU "\t[%s] " GRN Fmt RESET, __FUNCTION__, ##__VA_ARGS__); }
#define MYDEBUG_1(Fmt, ...) \
  { printf(GRN Fmt RESET, ##__VA_ARGS__); }
#define MYPRINT(Fmt, ...) \
  { printf(CYN Fmt RESET, ##__VA_ARGS__); }

inline std::string make_error_log(std::string msg, char const *file,
                                  char const *function, std::size_t line) {
  return std::string{"\n\n" RED} + file + "(" + std::to_string(line) + "): [" +
         function + "] \n\t" + msg + "\n\n" RESET;
}
#define ERROR_LOG(...) make_error_log(__VA_ARGS__, __FILE__, __func__, __LINE__)

#define GTUCKER_REMOVE_POINTER_TYPE_ALIAS(Type) \
  typename std::remove_pointer<Type>::type

template <typename T>
T *allocate(size_t num) {
  T *ptr = static_cast<T *>(malloc(sizeof(T) * num));
  if (ptr == NULL) {
    throw std::runtime_error(
        std::string("Memory Allocation ERROR \n\t [ptr == NULL]"));
  }
  return ptr;
}

template <typename T>
void deallocate(T *ptr) {
  free(ptr);
}

template <typename T>
T frand(T x, T y) {
  return ((y - x) * (static_cast<T>(rand()) / RAND_MAX)) + x;
}  // return the random value in (x, y) interval

template <typename T>
T abs(T x) {
  return x > 0 ? x : -x;
}

/* helper */
#define _ERROR_TO_STRING(err) \
  case err:                   \
    return #err;
inline const char *_cuda_get_error_enum(cudaError_t err) {
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

inline void _cuda_check(cudaError_t result, char *const func,
                        const char *const file, int const line) {
  if (result) {
    std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file,
                 line, static_cast<unsigned int>(result),
                 _cuda_get_error_enum(result), func);
    cudaDeviceReset();
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_API_CALL(val) _cuda_check((val), #val "", __FILE__, __LINE__)

}  // namespace gputucker
}  // namespace supertensor
#endif  // HELPER_HPP_