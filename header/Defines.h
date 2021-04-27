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
#else
#define DLL_API
#endif

// CASE Xcode C++ compiler
#ifdef __llvm__
#ifdef _DLL_EXPORTS
/* The classes below are exported */
#define MACOS_API_BEGIN _Pragma("GCC visibility push(default)")
#define MACOS_API_END _Pragma("GCC visibility pop")
#else
#define MACOS_API_BEGIN
#define MACOS_API_END
#endif
#else
#define MACOS_API_BEGIN
#define MACOS_API_END
#endif
