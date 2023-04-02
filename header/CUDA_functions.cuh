#pragma once
#include "Defines.cuh"
#ifndef CUBLAS_NDEBUG
#include <cublas_v2.h>
#endif

#ifndef CUBLAS_NDEBUG
/*-----------------------------------------------------------*/
/*             CUBLAS ERROR MESSAGES ENUMERATOR              */
/*-----------------------------------------------------------*/
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";

	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";

	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "<unknown>";
}

/*-----------------------------------------------------------*/
/*                   CUBLAS ERROR CHECKING                   */
/*-----------------------------------------------------------*/
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d, error: %s\nterminating!\n", __FILE__, __LINE__, \
			_cublasGetErrorEnum(err)); \
			assert(0); \
	}
}

/*-----------------------------------------------------------*/
/*               CUBLAS ERROR CHECKING (macro)               */
/*-----------------------------------------------------------*/
#define cublascheck(ans) { __cublasSafeCall((ans), __FILE__, __LINE__); }
#endif

namespace CUDA_GPU {

CUDA_API_BEGIN
	class CUDA_DLL_API Kernels {


	public:

		int N_rows; // number of rows
		int N_cols; // number of rows
		int N_grids; // number of rows
		int N_actual; // actual number of elements

		size_t bytes;

		float* atoms_x_h; // x vector (host)
		float* atoms_y_h; // y vector (host)
		float* atoms_x_d; // x vector (device)
		float* atoms_y_d; // y vector (device)

		float* diffs_x_d;
		float* diffs_y_d;
		float* force_x_h; // Force x vector (host)
		float* force_y_h; // Force y vector (host)
		float* force_x_d; // Force x vector (device)
		float* force_y_d; // Force y vector (device)
		float* d_ones_force; // vector of ones to multiply matrix with (device)
		float* d_ones_track; // vector of ones to multiply matrix with (device)

		size_t bytes_rows; // Size, in bytes, of each vector
		size_t bytes_cols; // Size, in bytes, of each vector

		float* G_h; // tracing matrix (host)
		float* G_d; // tracing matrix (device)
		float* G_track_h; // tracking matrix (host)
		float* G_track_d; // tracking matrix (device)

		float* p_d; // percentage vector (device)
		float* p_h; // percentage vector (host)

		const float value = 1.f;
		float alpha;
		float beta;

		int threads_per_block;

#ifndef CUBLAS_NDEBUG
		cublasHandle_t local_handle;
#endif
		/*-----------------------------------------------------------*/
		/*						 Constructor					     */
		/*-----------------------------------------------------------*/
		Kernels(const int n_pop, const int n_grids, const int threads_per_block_in, const cublasHandle_t handle);

		/*-----------------------------------------------------------*/
		/*              Repulsive force evaluation (GPU)             */
		/*-----------------------------------------------------------*/
		void pairwise_gpu(Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y, float SD_factor);

		/*-----------------------------------------------------------*/
		/*              Repulsive force evaluation (GPU)             */
		/*-----------------------------------------------------------*/
		void tracker_gpu(Eigen::ArrayXf atoms_x, Eigen::ArrayXf atoms_y);

		/*-----------------------------------------------------------*/
		/*				 Percentage vector (getter)				     */
		/*-----------------------------------------------------------*/
		void get_p(Eigen::ArrayXf *p);

		/*-----------------------------------------------------------*/
		/*					Force vectors (getter)				     */
		/*-----------------------------------------------------------*/
		void get_forces(Eigen::ArrayXf *force_x, Eigen::ArrayXf *force_y);

		/*-----------------------------------------------------------*/
		/*					 Tracking matrix (getter)			     */
		/*-----------------------------------------------------------*/
		void get_G(Eigen::ArrayXXf *G);

		/*-----------------------------------------------------------*/
		/*					 Tracking matrix (getter)			     */
		/*-----------------------------------------------------------*/
		void get_G_trace(Eigen::ArrayXXf *G_track);

		/*-----------------------------------------------------------*/
		/*						 Destructor						     */
		/*-----------------------------------------------------------*/
		~Kernels();

	};
CUDA_API_END

}