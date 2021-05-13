#pragma once

#include "nomad.hpp"
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
	bool eval_x(NOMAD::Eval_Point &x, const NOMAD::Double &h_max, bool &count_eval) const;

	/*----------------------------------------*/
	/*            Success callback            */
	/*----------------------------------------*/
	void update_success(const NOMAD::Stats &stats, const NOMAD::Eval_Point &x);

	// Model parameters
	int healthcare_capacity; // healthcare capacity
	int eval_k; // number of samples for estimates
	NOMAD::Point lb; // get lower bounds
	NOMAD::Point ub; // get upper bounds

	std::string log_file; // log filename
	std::string feasible_file; // feasible history filename
	std::string infeasible_file; // infeasible history filename

	NOMAD::Mads* session;

	double **history; // an array of int arrays (a pointer to pointers to ints)
	size_t row_size;
	size_t n_rows;
	int *n_evals;
	int *bbe;

};

#endif // MY_EVALUATOR_SINGLEOBJ_H