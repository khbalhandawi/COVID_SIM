/*-----------------------------------------------------*/
/*   use the NOMAD library with a COVID user function  */
/*-----------------------------------------------------*/
#include "NOMAD_Evaluator_singleobj.h"
#include "io_blackbox_functions.h"
#include "io_utilities.h"

#include "nomad.hpp"
#include <iterator>
using namespace std;

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main(int argc, char ** argv) {

	int n_sargs = 6; // number of static arguments

	// set default arguments if input arguments not provided
	int healthcare_capacity = 90;
	int eval_k = 20;
	std::string log_file = "NOMAD_hist.txt";
	std::string feasible_file = "f_hist_NOMAD.txt";
	std::string infeasible_file = "i_hist_NOMAD.txt";
	int nb_proc = 2;

	if (argc == n_sargs + 1) {
		healthcare_capacity = stoi(argv[1]);
		eval_k = stoi(argv[2]);
		log_file = argv[3];
		feasible_file = argv[4];
		infeasible_file = argv[5];
		nb_proc = stoi(argv[6]);
	}

	// display:
	NOMAD::Display out(std::cout);
	out.precision(NOMAD::DISPLAY_PRECISION_STD);

	try {

		// NOMAD initializations:
		NOMAD::begin(argc, argv);

		// parameters creation:
		NOMAD::Parameters p(out);

		p.set_DIMENSION(3);             // number of variables

		std::vector<NOMAD::bb_output_type> bbot(2); // definition of
		bbot[0] = NOMAD::OBJ;                   // output types
		bbot[1] = NOMAD::PB;
		p.set_BB_OUTPUT_TYPE(bbot);

		//p.set_DISPLAY_ALL_EVAL(true);   // displays all evaluations.
		p.set_DISPLAY_STATS("bbe ( sol ) obj");

		// initial point
		NOMAD::Point x0(3);
		x0[0] = 0.5;   // n_violators
		x0[1] = 0.5;   // SD
		x0[2] = 0.5;   // testing
		p.set_X0(x0);

		// actual bounds for unscaling
		NOMAD::Point lb_us(3), ub_us(3);
		lb_us[0] = 16.0; ub_us[0] = 101.0;  // n_violators
		lb_us[1] = 0.0001; ub_us[1] = 0.15; // SD
		lb_us[2] = 10.0; ub_us[2] = 51.0;   // testing

		// bounds
		NOMAD::Point lb(3), ub(3);
		lb[0] = 0.0; ub[0] = 1.0;
		lb[1] = 0.0; ub[1] = 1.0;
		lb[2] = 0.0; ub[2] = 1.0;                        
		p.set_LOWER_BOUND(lb);
		p.set_UPPER_BOUND(ub);

		p.set_MAX_BB_EVAL(10000 / eval_k);     // the algorithm terminates after 500 black-box evaluations

		p.set_DISPLAY_DEGREE(2);
		p.set_SOLUTION_FILE("sol.txt");

		// parameters validation:
		p.check();

		// Print titles to file
		std::vector<std::string> titles(8);
		titles = { "bbe","x1","x2","x3","f","cstr","p_value" };

		ofstream output_file, f_file, i_file;
		output_file.open(log_file, ofstream::out);
		IO_BB::writeToFile<std::string>(titles, &output_file);
		output_file.close();
		f_file.open(feasible_file, ofstream::out);
		IO_BB::writeToFile<std::string>(titles, &f_file);
		f_file.close();
		i_file.open(infeasible_file, ofstream::out);
		IO_BB::writeToFile<std::string>(titles, &i_file);
		i_file.close();

		// custom evaluator creation:
		My_Evaluator ev(p, 10000, 7, nb_proc);
		ev.healthcare_capacity = healthcare_capacity;
		ev.eval_k = eval_k; // number of samples for estimates
		ev.lb = lb_us; // get lower bounds
		ev.ub = ub_us; // get upper bounds
		ev.log_file = log_file; // log filename
		ev.feasible_file = feasible_file; // feasible history filename
		ev.infeasible_file = infeasible_file; // infeasible history filename

		// algorithm creation and execution:
		NOMAD::Mads mads(p, &ev);
		ev.session = &mads; // reference mads stats inside evaluator
		mads.run();
	}
	catch (exception & e) {
		cerr << "\nNOMAD has been interrupted (" << e.what() << ")\n\n";
	}

	NOMAD::Slave::stop_slaves(out);
	NOMAD::end();

	return EXIT_SUCCESS;
}