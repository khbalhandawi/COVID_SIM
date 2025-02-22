
#include "NOMAD_Evaluator_singleobj.h"
#include "io_blackbox_functions.h"
#include "io_utilities.h"

#include <vector>
#include <iterator>
#include <iomanip>
#include <math.h>
#include <omp.h>

/*----------------------------------------------------*/
/*                         eval_x                     */
/*----------------------------------------------------*/
bool My_Evaluator::eval_x(NOMAD::Eval_Point   &x,
	const NOMAD::Double &h_max,
	bool         &count_eval)
{
	NOMAD::Double f, g1; // objective function

	//--------------------------------------------------------------//
	// Read from data file

	count_eval = false;

	double SD;
	int run, sample, n_violators, test_capacity;

	COVID_SIM::Configuration Config;

	/*-----------------------------------------------------------*/
	/*                Read command line arguments                */
	/*-----------------------------------------------------------*/

	// Model variables
	double x1, x2, x3;

	x1 = x[0].value(); // get x1
	x2 = x[1].value(); // get x1
	x3 = x[2].value(); // get x1

	n_violators = std::round(IO_BB::scaling(x1, lb[0].value(), ub[0].value(), 2)); // should be rounded to nearest int
	SD = IO_BB::scaling(x2, lb[1].value(), ub[1].value(), 2);
	test_capacity = std::round(IO_BB::scaling(x3, lb[2].value(), ub[2].value(), 2)); // should be rounded to nearest int

	// Display input arguments
	//std::cout << "\n" << "================= starting =================" << std::endl;
	//std::cout << "E: " << n_violators << " | SD: " << SD << " | T: " << test_capacity << " | H_c: " << healthcare_capacity << " | output: " << log_file << "\n";

	/*-----------------------------------------------------------*/
	/*            Simulation configuration variables             */
	/*-----------------------------------------------------------*/
	// initialize
	const char config_file[] = "configuration.ini";

	COVID_SIM::load_config(&Config, config_file);
	Config.set_from_file();

	//std::cout << "Config loaded!" << std::endl;

	/*-----------------------------------------------------------*/
	/*                      Design variables                     */
	/*-----------------------------------------------------------*/

	run = 0;
	Config.social_distance_factor = 1e-6 * SD * Config.force_scaling;
	Config.social_distance_violation = n_violators; // number of people
	Config.healthcare_capacity = healthcare_capacity;
	Config.number_of_tests = test_capacity;
	Config.log_file = log_file;
	Config.run_i = run;

	// Initialize progress files
	if (*bbe == 0) {
		int new_bbe; double mean_f, mean_c, p_value;
		parallelCompute(Config, run, eval_k_success, new_bbe, mean_f, mean_c, p_value);

		//====================================================//
		// Print results to file

		ofstream f_file, i_file;

		// _p is a protected member of Evaluator
		f_file.open(feasible_success_file, ofstream::app);
		i_file.open(infeasible_success_file, ofstream::app);

		/*---------------------- Output log ----------------------*/
		std::vector<double> matrix_out; // unstripped matrix
		matrix_out.push_back(*n_successes_f);
		matrix_out.push_back(*bbe);
		matrix_out.push_back(x1);
		matrix_out.push_back(x2);
		matrix_out.push_back(x3);
		matrix_out.push_back(mean_f);
		matrix_out.push_back(mean_c);
		matrix_out.push_back(p_value);

		IO_BB::writeToFile<double>(matrix_out, &f_file);
		IO_BB::writeToFile<double>(matrix_out, &i_file);
		f_file.close();

		*n_successes_f += 1;
		*n_successes_i += 1;
	}

	int new_bbe; double mean_f, mean_c, p_value;
	parallelCompute(Config, run, eval_k, new_bbe, mean_f, mean_c, p_value);

	x.set_bb_output(0, mean_f);
	x.set_bb_output(1, mean_c);

	// convert eval point to int vector
	double coor_i;
	NOMAD::Double coor;

	history[*n_evals][0] = x.get_tag();
	for (size_t k = 0; k < x.size(); k++) {
		coor = x.get_coord(int(k));
		coor_i = coor.value();
		history[*n_evals][1 + k] = coor_i;
	}
	history[*n_evals][1 + x.size()] = mean_f;
	history[*n_evals][2 + x.size()] = mean_c;
	history[*n_evals][3 + x.size()] = p_value;

	// display history
	//for (int i = 0; i < *n_evals + 1; i++) {
	//	for (int j = 0; j < row_size; j++) {
	//		cout << history[i][j] << ", ";
	//	}
	//	cout << endl;
	//}

	count_eval = true;
	//====================================================//
	// Get best feasible and infeasible points

	const NOMAD::Eval_Point *bf, *bi;
	bf = session->get_best_feasible();
	bi = session->get_best_infeasible();
	//====================================================//
	// Print results to file

	ofstream output_file, f_file, i_file;
	output_file.open(log_file, ofstream::app);
	f_file.open(feasible_file, ofstream::app);
	i_file.open(infeasible_file, ofstream::app);

	for (sample = 0; sample < eval_k; sample++) {
		/*---------------------- Output log ----------------------*/
		std::vector<double> matrix_out; // unstripped matrix
		matrix_out.push_back(*bbe + sample + 1);
		matrix_out.push_back(x1);
		matrix_out.push_back(x2);
		matrix_out.push_back(x3);
		matrix_out.push_back(mean_f);
		matrix_out.push_back(mean_c);
		matrix_out.push_back(p_value);

		IO_BB::writeToFile<double>(matrix_out, &output_file);

		/*---------------------- Output log ----------------------*/
		std::vector<double> bf_vec, bi_vec;
		//bbe, x_infeas', f_feas, c_feas, p_feas, h_feas
		if (bf) {
			bf_vec = IO_BB::lookupHistory(bf->get_tag(), *n_evals, history);
			bf_vec[0] = *bbe + sample + 1;
		}
		else {
			bf_vec = IO_BB::lookupHistory(0, *n_evals, history); // get the intial point data
			bf_vec[0] = *bbe + sample + 1;
		}

		if (bi) {
			bi_vec = IO_BB::lookupHistory(bi->get_tag(), *n_evals, history);
			bi_vec[0] = *bbe + sample + 1;
		}
		else {
			bi_vec = IO_BB::lookupHistory(0, *n_evals, history); // get the intial point data
			bi_vec[0] = *bbe + sample + 1;
		}

		IO_BB::writeToFile<double>(bf_vec, &f_file);
		IO_BB::writeToFile<double>(bi_vec, &i_file);

	}

	output_file.close();
	f_file.close();
	i_file.close();

	*n_evals += 1;
	*bbe = new_bbe;

	return true;
}

/*----------------------------------------------------*/
/*                   update_success                   */
/*----------------------------------------------------*/
void My_Evaluator::update_success(const NOMAD::Stats &stats, const NOMAD::Eval_Point &x)
{
	// User updates after a success.
	/**
	 This virtual method is called every time a new (full) success is made.
	 \param stats Stats                 -- \b IN.
	 \param x     Last successful point -- \b IN.
	 */

	//--------------------------------------------------------------//
	// Read from data file

	double SD;
	int run, sample, n_violators, test_capacity;

	COVID_SIM::Configuration Config;

	/*-----------------------------------------------------------*/
	/*                Read command line arguments                */
	/*-----------------------------------------------------------*/

	// Model variables
	double x1, x2, x3;

	x1 = x[0].value(); // get x1
	x2 = x[1].value(); // get x1
	x3 = x[2].value(); // get x1

	n_violators = std::round(IO_BB::scaling(x1, lb[0].value(), ub[0].value(), 2)); // should be rounded to nearest int
	SD = IO_BB::scaling(x2, lb[1].value(), ub[1].value(), 2);
	test_capacity = std::round(IO_BB::scaling(x3, lb[2].value(), ub[2].value(), 2)); // should be rounded to nearest int

	/*-----------------------------------------------------------*/
	/*            Simulation configuration variables             */
	/*-----------------------------------------------------------*/
	// initialize
	const char config_file[] = "configuration.ini";

	COVID_SIM::load_config(&Config, config_file);
	Config.set_from_file();

	/*-----------------------------------------------------------*/
	/*                      Design variables                     */
	/*-----------------------------------------------------------*/

	run = 0;
	Config.social_distance_factor = 1e-6 * SD * Config.force_scaling;
	Config.social_distance_violation = n_violators; // number of people
	Config.healthcare_capacity = healthcare_capacity;
	Config.number_of_tests = test_capacity;
	Config.log_file = log_file;
	Config.run_i = run;

	int new_bbe; double mean_f, mean_c, p_value;
	parallelCompute(Config, run, eval_k_success, new_bbe, mean_f, mean_c, p_value);

	//====================================================//
	// Print results to file

	ofstream file;
	int n_successes;

	// _p is a protected member of Evaluator
	if (x.is_feasible(_p.get_h_min())) {
		file.open(feasible_success_file, ofstream::app);
		n_successes = *n_successes_f;
	}
	else {
		file.open(infeasible_success_file, ofstream::app);
		n_successes = *n_successes_i;
	}

	/*---------------------- Output log ----------------------*/
	std::vector<double> matrix_out; // unstripped matrix
	matrix_out.push_back(n_successes);
	matrix_out.push_back(*bbe);
	matrix_out.push_back(x1);
	matrix_out.push_back(x2);
	matrix_out.push_back(x3);
	matrix_out.push_back(mean_f);
	matrix_out.push_back(mean_c);
	matrix_out.push_back(p_value);

	IO_BB::writeToFile<double>(matrix_out, &file);
	file.close();

	if (x.is_feasible(_p.get_h_min())) {
		*n_successes_f += 1;
	}
	else {
		*n_successes_i += 1;
	}
	
	
}

/*----------------------------------------------------*/
/*                Parallel Simulation                 */
/*----------------------------------------------------*/
void My_Evaluator::parallelCompute(const COVID_SIM::Configuration &Config, const int &run, const int &n_samples, int &new_bbe, double &mean_f, double &mean_c, double &p_value)
{
	// Global variables
	int sample;
	double* objs;
	double* cstrs;
	objs = new double[n_samples];
	cstrs = new double[n_samples];

	// Initialize simulation objects for sampling
	COVID_SIM::simulation** sims;
	sims = new COVID_SIM::simulation*[n_samples];
	for (int i = 0; i < n_samples; i++) {

		// seed random generator
		/* using nano-seconds instead of seconds */
		unsigned long seed = static_cast<uint32_t>(chrono::high_resolution_clock::now().time_since_epoch().count());

		// run the simulation while loop without QT
#ifdef GPU_ACC
		sims[i] = new COVID_SIM::simulation(Config, seed, handle);
#else
		sims[i] = new COVID_SIM::simulation(Config, seed);
#endif

	}

	// Parallel for loop over samples
	//#pragma omp parallel firstprivate(sims)
	#pragma omp parallel 
	{
		// private variables
		std::vector<double> matrix_opt;
		double infected, fatalities, mean_distance, mean_GC, obj_1, obj_2, c1;

		#pragma omp for schedule(static, 4)
		for (sample = 0; sample < n_samples; sample++) {

			/*-----------------------------------------------------------*/
			/*                        Run blackbox                       */
			/*-----------------------------------------------------------*/

			// run the simulation while loop without QT
			//std::cout << "initialized simulation" << std::endl;
			matrix_opt = COVID_SIM::processInput(run, sims[sample]);

			/*-----------------------------------------------------------*/
			/*                    Log blackbox outputs                   */
			/*-----------------------------------------------------------*/

			// Output evaluation to file
			infected = matrix_opt[0];
			fatalities = matrix_opt[1];
			mean_distance = matrix_opt[2];
			mean_GC = matrix_opt[3];

			obj_1 = -mean_GC;
			obj_2 = fatalities;
			c1 = infected - healthcare_capacity;

			//cout << "thread_id: " << omp_get_thread_num() << " | obj: " << obj_1 << endl;
			//#pragma omp critical 
			{
				objs[sample] = obj_1;
				cstrs[sample] = c1;
			}


		}
	}

	// Free memory allocated to simulation objects
	for (int i = 0; i < n_samples; i++) {
		delete sims[i];
	}
	delete sims;

	new_bbe = *bbe + n_samples;
	mean_f = IO_BB::average(objs, n_samples);
	mean_c = IO_BB::average(cstrs, n_samples);
	p_value = IO_BB::reliability(cstrs, n_samples);

	delete objs, cstrs;
}

/*----------------------------------------------------*/
/*                     Constructor                    */
/*----------------------------------------------------*/
My_Evaluator::My_Evaluator(const NOMAD::Parameters &p, int max_bb_eval, int length_history, int nb_proc) : Evaluator(p)
{
	row_size = length_history;
	n_rows = max_bb_eval;
	omp_set_num_threads(nb_proc); // Use 4 threads for all consecutive parallel regions

	// allocate an array of N pointers to ints
	// malloc returns the address of this array (a pointer to (int *)'s)
	history = (double **)malloc(sizeof(double *)*n_rows);
	// for each row, malloc space for its buckets and add it to the array of arrays
	for (int i = 0; i < n_rows; i++) {
		history[i] = (double *)malloc(sizeof(double)*row_size);
	}
	n_evals = new int; *n_evals = 0;
	n_successes_f = new int; *n_successes_f = 1;
	n_successes_i = new int; *n_successes_i = 1;

	bbe = new int;
	*bbe = 0;

#ifdef GPU_ACC
	cublascheck(cublasCreate(&handle)); // construct cublas handle
#endif
}

/*----------------------------------------------------*/
/*                     Destructor                     */
/*----------------------------------------------------*/
My_Evaluator::~My_Evaluator(void) 
{
	free(history);
	delete n_evals, bbe;
#ifdef GPU_ACC
	cublascheck(cublasDestroy(handle)); // destroy cublas handle to avoid malloc errors
#endif
}



