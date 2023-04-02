#pragma once

#define _DLL_EXPORTS

// CASE Visual Studio C++ compiler
#ifdef _MSC_VER
#pragma warning(disable:4251)
#ifdef _DLL_EXPORTS
#define DLL_API __declspec(dllexport) 
#else
#define DLL_API __declspec(dllimport)
#endif
#define GPU_ACC // comment to disable GPU acceleration
// CASE GCC compiler
#elif __GNUC__
#ifdef _DLL_EXPORTS
/* The classes below are exported */
// #define API_BEGIN extern "C" {
// #define API_END }
#define API_BEGIN extern "C"
#define API_END
#define DLL_API __attribute__ ((visibility ("default")))
#else
#define API_BEGIN
#define API_END
#define DLL_API
#endif
#define GPU_ACC // comment to disable GPU acceleration
// CASE Xcode C++ compiler
#elif __llvm__
#define API_BEGIN
#define API_END
#ifdef _DLL_EXPORTS
/* The classes below are exported */
#define API_BEGIN _Pragma("GCC visibility push(default)")
#define API_END _Pragma("GCC visibility pop")
#define DLL_API
#else
#define API_BEGIN
#define API_END
#define DLL_API
#endif
// #define GPU_ACC // comment to disable GPU acceleration
#else
#define DLL_API
#define API_BEGIN
#define API_END
#endif