#pragma once

// CASE Visual Studio C++ compiler
#ifdef _MSC_VER
#pragma warning(disable:4251)
#ifdef _GPU_DLL_EXPORTS
#define CUDA_DLL_API __declspec(dllexport) 
#else
#define CUDA_DLL_API __declspec(dllimport) 
#endif
#else
#define DLL_API
#endif

//#define CUBLAS_NDEBUG
//#define RANDOM_DEBUG