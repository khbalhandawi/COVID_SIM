/*---------------------------------------------------------------------------------*/
/*  COVID SIM - Agent-based model of a pandemic in a population -                  */
/*                                                                                 */
/*  COVID SIM - version 1.0.0 has been created by                                  */
/*                 Khalil Al Handawi           - McGill University                 */
/*                                                                                 */
/*  The copyright of NOMAD - version 3.9.1 is owned by                             */
/*                 Khalil Al Handawi           - McGill University                 */
/*                 Michael Kokkolaras          - McGill University                 */
/*                                                                                 */
/*                                                                                 */
/*  Contact information:                                                           */
/*    McGill University - Systems Optimization Lab (SOL)                           */
/*    Macdonald Engineering Building, 817 Sherbrooke Street West,                  */
/*    Montreal (Quebec) H3A 0C3 Canada                                             */
/*    e-mail: khalil.alhandawi@mail.mcgill.ca                                      */
/*    phone : 1-514-398-2343                                                       */
/*                                                                                 */
/*  This program is free software: you can redistribute it and/or modify it        */
/*  under the terms of the GNU Lesser General Public License as published by       */
/*  the Free Software Foundation, either version 3 of the License, or (at your     */
/*  option) any later version.                                                     */
/*                                                                                 */
/*  This program is distributed in the hope that it will be useful, but WITHOUT    */
/*  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or          */
/*  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License    */
/*  for more details.                                                              */
/*                                                                                 */
/*  You should have received a copy of the GNU Lesser General Public License       */
/*  along with this program. If not, see <http://www.gnu.org/licenses/>.           */
/*                                                                                 */
/*---------------------------------------------------------------------------------*/

/**
 \file   infection.cpp
 \brief  Functions for computing new infections, recoveries, and deaths (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    infection.h
 */

#include "infection.h"

 /*-----------------------------------------------------------*/
 /*                     Finds nearby IDs                      */
 /*-----------------------------------------------------------*/
void find_nearby(Eigen::ArrayXXf population, vector<double> infection_zone, Eigen::ArrayXf &indices, int &infected_number, 
	bool traveling_infects, string kind, Eigen::ArrayXXf infected_previous_step)
{
	/*finds nearby IDs

	Keyword Arguments
	---------------- -

	kind : str(can be 'healthy' or 'infected')
	determines whether infected or healthy individuals are returned
	within the infection_zone


	Returns
	------ -
	if kind = 'healthy', indices of healthy agents within the infection
	zone is returned.This is because for each healthy agent, the chance to
	become infected needs to be tested

	if kind = 'infected', only the number of infected within the infection zone is
	returned.This is because in this situation, the odds of the healthy agent at
	the center of the infection zone depend on how many infectious agents are around
	it.
	*/

	if (kind == "healthy") {
		ArrayXXb cond(population.rows(), 5);

		cond << (infection_zone[0] < population.col(1)),
				(population.col(1) < infection_zone[2]),
				(infection_zone[1] < population.col(2)),
				(population.col(2) < infection_zone[3]),
				(population.col(6) == 0);


		indices = population(select_rows(cond), { 0 });
	}
	else if (kind == "infected") {
		if (traveling_infects) {
			ArrayXXb cond(infected_previous_step.rows(), 5);

			cond << (infection_zone[0] < infected_previous_step.col(1)),
					(infected_previous_step.col(1) < infection_zone[2]),
					(infection_zone[1] < infected_previous_step.col(2)),
					(infected_previous_step.col(2) < infection_zone[3]),
					(infected_previous_step.col(6) == 1);

			infected_number = infected_previous_step(select_rows(cond), { 6 }).rows();
		}
		else {
			ArrayXXb cond(infected_previous_step.rows(), 6);

			cond << (infection_zone[0] < infected_previous_step.col(1)),
					(infected_previous_step.col(1) < infection_zone[2]),
					(infection_zone[1] < infected_previous_step.col(2)),
					(infected_previous_step.col(2) < infection_zone[3]),
					(infected_previous_step.col(6) == 1),
					(infected_previous_step.col(11) == 0);

			infected_number = infected_previous_step(select_rows(cond), { 6 }).rows();
		}

	}
}

/*-----------------------------------------------------------*/
/*                    Find new infections                    */
/*-----------------------------------------------------------*/
void infect(Eigen::ArrayXXf &population, Eigen::ArrayXXf &destinations, 
	Configuration Config, int frame, RandomDevice *my_rand, bool send_to_location, 
	vector<double> location_bounds, int location_no, double location_odds, bool test_flag)
{
	/*finds new infections.

	Function that finds new infections in an area around infected persens
	defined by infection_range, and infects others with chance infection_chance

	Keyword arguments
	---------------- -
	population : ndarray
	array containing all data on the population

	pop_size : int
	the number if individuals in the population

	infection_range : float
	the radius around each infected person where transmission of virus can take place

	infection_chance : float
	the odds that the virus infects someone within range(range 0 to 1)

	frame : int
	the current timestep of the simulation

	healthcare_capacity : int
	the number of places available in the healthcare system

	verbose : bool
	whether to report illness events

	send_to_location : bool
	whether to give infected people a destination

	location_bounds : list
	the location bounds where the infected person is sent to and can roam
	within(xmin, ymin, xmax, ymax)

	destinations : list or ndarray
	the destinations vector containing destinations for each individual in the population.
	Needs to be of same length as population

	location_no : int
	the location number, used as index for destinations array if multiple possible
	destinations are defined

	location_odds : float
	the odds that someone goes to a location or not.Can be used to simulate non - compliance
	to for example self - isolation.

	traveling_infects : bool
	whether infected people heading to a destination can still infect others on the way there

	// TODO: Convert infection range to a circle based on pairwise dist matrix

	*/

	// mark those already infected first
	Eigen::ArrayXXf infected_previous_step = population(select_rows(population.col(6) == 1), Eigen::all);
	Eigen::ArrayXXf healthy_previous_step = population(select_rows(population.col(6) == 0), Eigen::all);

	vector<int> new_infections;
	Eigen::ArrayXf patient, person;
	vector<double> infection_zone;
	Eigen::ArrayXf indices;
	int infected_number = 0;

	// if less than half are infected, slice based on infected (to speed up computation)
	if ( infected_previous_step.rows() < (floor(Config.pop_size / 2)) ) {
		for (int i = 0; i < infected_previous_step.rows(); i++) {
			patient = infected_previous_step.row(i);
			// define infection zone for patient
			infection_zone = { patient[1] - Config.infection_range, patient[2] - Config.infection_range,
				patient[1] + Config.infection_range, patient[2] + Config.infection_range }; // Its a square region

			// find healthy people surrounding infected patient
			if ( (Config.traveling_infects) || (patient[11] == 0) ) {
				find_nearby(population, infection_zone, indices, infected_number, false, "healthy");
			}

			for (auto i : indices) {
				// roll die to see if healthy person will be infected
				if (my_rand->rand() < Config.infection_chance) {
					population.block(i, 6, 1, 1) = 1;
					population.block(i, 8, 1, 1) = frame;
					new_infections.push_back(i);
				}

			}

		}

	}
	else {
		// if more than half are infected slice based in healthy people (to speed up computation)
		for (int i = 0; i < healthy_previous_step.rows(); i++) {
			person = healthy_previous_step.row(i);
			// define infecftion range around healthy person
			infection_zone = { person[1] - Config.infection_range, person[2] - Config.infection_range,
				person[1] + Config.infection_range, person[2] + Config.infection_range }; // Its a square region

			// if person is not already infected, find if infected are nearby
			if (person[6] == 0) {
				// find infected nearby healthy person (infected_number = poplen)
				if (Config.traveling_infects) {
					find_nearby(population, infection_zone, indices, infected_number, true, "infected");
				}
				else {
					find_nearby(population, infection_zone, indices, infected_number, true, "infected", infected_previous_step);
				}

				if (infected_number > 0) {
					if (my_rand->rand() < Config.infection_chance * infected_number) {
						// roll die to see if healthy person will be infected
						population.block(person[0], 6, 1, 1) = 1;
						population.block(person[0], 8, 1, 1) = frame;
						new_infections.push_back(person[0]);

					}
				}

			}

		}
	}


	if ((send_to_location) && (test_flag)) {
		ArrayXXb cond(Config.pop_size, 3);

		// randomly pick individuals for testing
		Eigen::ArrayXXf inside_world = population(select_rows(population.col(11) == 0), Eigen::all);
		int n_samples = min(Config.number_of_tests, int(inside_world.rows()));

		// flag these individuals for testing
		Eigen::VectorXi Choices = my_rand->Random_choice(inside_world.col(0), n_samples);
		population(Choices, { 18 }) = 1;

		// condition for testing
		cond << (population.col(18) == 1), (population.col(6) == 1), ((frame - population.col(8)) >= Config.incubation_period);

		//=================================================================//
		// People that need to be hospitalized (decide who gets care randomly)
		
		if (select_rows(cond).size() > 0) {

			population(Choices, 18) = 0; // reset testing flags
			Eigen::ArrayXXf tested_pop = population(select_rows(cond), Eigen::all);
			Eigen::ArrayXXf hospitalized_pop = population(select_rows(population.col(10) == 1), Eigen::all);

			int room_left = (Config.healthcare_capacity - hospitalized_pop.rows());
			n_samples = min(int(tested_pop.rows()), room_left);

			// flag these individuals for hospitalization following testing
			Choices = my_rand->Random_choice(tested_pop.col(0), n_samples);
			population(Choices, { 18 }) = 1;

			// hospitalize sick individuals
			population(Choices, { 10 }) = 1;

			go_to_location(Choices, population, destinations, location_bounds, location_no);
		}
	}

	population.col(18) = 0; // reset testing flag

	if ((new_infections.size() > 0) && (Config.verbose) && (Config.report_status)) {
		printf("\nat timestep %i these people got sick: %i", frame, new_infections.size());
	}
}

/*-----------------------------------------------------------*/
/*                     Recover or die                        */
/*-----------------------------------------------------------*/
void recover_or_die(Eigen::ArrayXXf &population, int frame, Configuration Config, RandomDevice *my_rand)
{
	/*see whether to recover or die

	Keyword arguments
	---------------- -
	population : ndarray
	array containing all data on the population

	frame : int
	the current timestep of the simulation

	recovery_duration : tuple
	lower and upper bounds of duration of recovery, in simulation steps

	mortality_chance : float
	the odds that someone dies in stead of recovers(between 0 and 1)

	risk_age : int or flaot
	the age from which mortality risk starts increasing

	critical_age : int or float
	the age where mortality risk equals critical_mortality_change

	critical_mortality_chance : float
	the heightened odds that an infected person has a fatal ending

	risk_increase : string
	can be 'quadratic' or 'linear', determines whether the mortality risk
	between the at risk age and the critical age increases linearly or
	exponentially

	no_treatment_factor : int or float
	defines a change in mortality odds if someone cannot get treatment.Can
	be larger than one to increase risk, or lower to decrease it.

	treatment_dependent_risk : bool
	whether availability of treatment influences patient risk

	treatment_factor : int or float
	defines a change in mortality odds if someone is in treatment.Can
	be larger than one to increase risk, or lower to decrease it.

	verbose : bool
	whether to report to terminal the recoveries and deaths for each simulation step
	*/

	// find infected people
	Eigen::ArrayXXf infected_people = population(select_rows(population.col(6) == 1), Eigen::all);

	// define vector of how long everyone has been sick
	Eigen::ArrayXf illness_duration_vector = frame - infected_people.col(8);

	Eigen::ArrayXf recovery_odds_vector = (illness_duration_vector - Config.recovery_duration[0]) / (Config.recovery_duration[1] - Config.recovery_duration[0]);
	recovery_odds_vector.max(0); // clip odds less than 0

	// update states of sick people
	Eigen::ArrayXf indices = infected_people(select_rows(recovery_odds_vector >= infected_people.col(9)), { 0 });

	vector<int> recovered;
	vector<int> fatalities;
	int age;
	double updated_mortality_chance;

	// decide whether to die or recover
	for (int idx : indices) {
	// check if we want risk to be age dependent
	// if age_dependent_risk:
		if (Config.age_dependent_risk) {

			age = infected_people(select_rows(infected_people.col(0) == idx), { 7 })[0];

			updated_mortality_chance = Config.mortality_chance;
			compute_mortality(age, updated_mortality_chance, Config.risk_age, Config.critical_age,
								   Config.critical_mortality_chance, Config.risk_increase);
		}
		else {
			updated_mortality_chance = Config.mortality_chance;
		}
		
		if ((infected_people(select_rows(infected_people.col(0) == idx), 10)[0] == 0) && (Config.treatment_dependent_risk)) {
			// if person is not in treatment, increase risk by no_treatment_factor
			updated_mortality_chance = updated_mortality_chance * Config.no_treatment_factor;
		}
		else if ((infected_people(select_rows(infected_people.col(0) == idx)[0], 10) == 1) && (Config.treatment_dependent_risk)) {
			// if person is in treatment, decrease risk by 
			updated_mortality_chance = updated_mortality_chance * Config.treatment_factor;
		}

		if (my_rand->rand() <= updated_mortality_chance) {
			// die
			infected_people(select_rows(infected_people.col(0) == idx)[0], 6) = 3;
			infected_people(select_rows(infected_people.col(0) == idx)[0], 10) = 0;
			fatalities.push_back(select_rows(infected_people.col(0) == idx)[0]);
		}
		else {
			// recover(become immune)
			infected_people(select_rows(infected_people.col(0) == idx)[0], 6) = 2;
			infected_people(select_rows(infected_people.col(0) == idx)[0], 10) = 0;
			recovered.push_back(select_rows(infected_people.col(0) == idx)[0]);
		}

	}

	if ((fatalities.size() > 0) && (Config.verbose) && (Config.report_status)) {
		printf("\nat timestep %i these people died: %i", frame, fatalities.size());
	}

	if ((recovered.size() > 0) && (Config.verbose) && (Config.report_status)) {
		printf("\nat timestep %i these people recovered: %i", frame, recovered.size());
	}

	// put array back into population
	population(select_rows(population.col(6) == 1), Eigen::all) = infected_people;

}

/*-----------------------------------------------------------*/
/*                     Compute mortality                     */
/*-----------------------------------------------------------*/
void compute_mortality(int age, double &mortality_chance, int risk_age,
	int critical_age, double critical_mortality_chance,
	string risk_increase)
{
	/*compute mortality based on age

	The risk is computed based on the age, with the risk_age marking
	the age where risk starts increasing, and the crticial age marks where
	the 'critical_mortality_odds' become the new mortality chance.

	Whether risk increases linearly or quadratic is settable.

	Keyword arguments
	---------------- -
	age : int
	the age of the person

	mortality_chance : float
	the base mortality chance
	can be very small but cannot be zero if increase is quadratic.

	risk_age : int
	the age from which risk starts increasing

	critical_age : int
	the age where mortality risk equals the specified
	critical_mortality_odds

	critical_mortality_chance : float
	the odds of dying at the critical age

	risk_increase : str
	defines whether the mortality risk between the at risk age
	and the critical age increases linearly or exponentially
	*/

	double step_increase;

	if ((age > risk_age) && (age < critical_age)) {
		// if age in range
		if (risk_increase == "linear") {
			// find linear risk
			step_increase = (critical_mortality_chance) / ((double(critical_age) - double(risk_age)) + 1);
			mortality_chance = critical_mortality_chance - ((double(critical_age) - double(age)) * step_increase);
		}
		else if (risk_increase == "quadratic") {
			// define exponential function between risk_age and critical_age
			int pw = 15;
			double A = exp(log(mortality_chance / critical_mortality_chance) / double(pw));
			double a = ((double(risk_age) - 1) - double(critical_age) * A) / (A - 1);
			double b = pow((mortality_chance / ((double(risk_age) - 1) + a)), pw);
			// define linespace
			Eigen::ArrayXf x = Eigen::ArrayXf::LinSpaced(critical_age, 0, critical_age);
			// find values
			Eigen::ArrayXf risk_values = (x + a).pow(pw) * b;
			mortality_chance = risk_values[age - 1];
		}
	}
	// if (age <= risk_age) simply return the base mortality chance
	else if (age >= critical_age) {
		//simply return the maximum mortality chance
		mortality_chance = critical_mortality_chance;
	}

}

/*-----------------------------------------------------------*/
/*              healthcare population infection              */
/*-----------------------------------------------------------*/
void healthcare_infection_correction(Eigen::ArrayXXf worker_population, RandomDevice *my_rand, double healthcare_risk_factor)
{
	/*corrects infection to healthcare population.

	Takes the healthcare risk factor and adjusts the sick healthcare workers
	by reducing(if < 0) ir increasing(if > 0) sick healthcare workers

	Keyword arguments
	---------------- -
	worker_population : ndarray
	the array containing all variables related to the healthcare population.
	Is a subset of the 'population' matrix.

	healthcare_risk_factor : int or float
	if other than one, defines the change in odds of contracting an infection.
		Can be used to simulate healthcare personell having extra protections in place(< 1)
		or being more at risk due to exposure, fatigue, or other factors(> 1)
	*/
	Eigen::ArrayXXf sick_workers;
	Eigen::ArrayXf cure_vector;

	if (healthcare_risk_factor < 0) {
		// set 1 - healthcare_risk_factor workers to non sick
		sick_workers = worker_population(select_rows(worker_population.col(6) == 1), Eigen::all);
		cure_vector = my_rand->uniform_dist(0.0, 1.0, sick_workers.rows(), 1);
		sick_workers(select_rows(cure_vector >= healthcare_risk_factor), { 6 }) = 0;
	}
	//else if (healthcare_risk_factor > 0) {
	//	// TODO : make proportion of extra workers sick
	//	pass
	//}
	//else {
	//	pass // if no changed risk, do nothing
	//}
	
	worker_population(select_rows(worker_population.col(6) == 1), Eigen::all) = sick_workers;
}