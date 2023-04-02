#pragma once

// CASE Visual Studio C++ compiler
#ifdef _MSC_VER
#pragma warning(disable:4251)
#ifdef _GPU_DLL_EXPORTS
#define CUDA_DLL_API __declspec(dllexport) 
#else
#define CUDA_DLL_API __declspec(dllimport) 
#endif
// CASE GCC compiler
#elif __GNUC__
// #define CUDA_API_BEGIN extern "C" {
// #define CUDA_API_END }
// #define CUDA_DLL_API
#define CUDA_API_BEGIN extern "C"
#define CUDA_API_END
#define CUDA_DLL_API __attribute__ ((visibility ("default")))
#else
#define CUDA_DLL_API
#define CUDA_API_BEGIN
#define CUDA_API_END
#endif

//#define CUBLAS_NDEBUG
//#define RANDOM_DEBUG