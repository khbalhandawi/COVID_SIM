#pragma once

#include "nomad.hpp"
#include "cublas_v2.h"
#include "simulation.h"
#include<vector>

#ifndef MY_EVALUATOR_SINGLEOBJ_H
#define MY_EVALUATOR_SINGLEOBJ_H

class My_Evaluator : public NOMAD::Evaluator
{
public:
	/*----------------------------------------*/
	/*               Constructor              */
	/*----------------------------------------*/
	My_Evaluator(const NOMAD::Parameters &p, int max_bb_eval, int length_history, int nb_proc);

	/*----------------------------------------*/
	/*               Destructor               */
	/*----------------------------------------*/
	~My_Evaluator(void);

	/*----------------------------------------*/
	/*               the problem              */
	/*----------------------------------------*/
	bool eval_x(NOMAD::Eval_Point &x, const NOMAD::Double &h_max, bool &count_eval);

	/*----------------------------------------*/
	/*            Success callback            */
	/*----------------------------------------*/
	void update_success(const NOMAD::Stats &stats, const NOMAD::Eval_Point &x);

	/*----------------------------------------*/
	/*          Parallel Simulation           */
	/*----------------------------------------*/
	void parallelCompute(const COVID_SIM::Configuration &Config, const int &run, const int &n_samples, int &new_bbe, double &mean_f, double &mean_c, double &p_value);


	// Model parameters
	int healthcare_capacity; // healthcare capacity
	int eval_k; // number of samples for estimates
	int eval_k_success; // number of samples for tracking progress
	NOMAD::Point lb; // get lower bounds
	NOMAD::Point ub; // get upper bounds

	std::string log_file; // log filename
	std::string feasible_file; // feasible history filename
	std::string infeasible_file; // infeasible history filename
	std::string feasible_success_file; // feasible progress filename
	std::string infeasible_success_file; // infeasible progress filename

	NOMAD::Mads* session;

	double **history; // an array of int arrays (a pointer to pointers to ints)
	size_t row_size;
	size_t n_rows;
	int *n_evals;
	int *bbe;
	int *n_successes_f;
	int *n_successes_i;

#ifdef GPU_ACC
	cublasHandle_t handle;
#endif

};

#endif // MY_EVALUATOR_SINGLEOBJ_H