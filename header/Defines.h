#pragma once

// CASE Visual Studio C++ compiler
#ifdef _MSC_VER
#pragma warning(disable:4251)
#ifdef _DLL_EXPORTS
#define DLL_API __declspec(dllexport) 
#else
#define DLL_API __declspec(dllexport) 
#endif
#else
#define DLL_API
#endif

#define GPU_ACC // comment to disable GPU acceleration