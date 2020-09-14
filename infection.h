#pragma once

#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <tuple>
#include "Configuration.h"
#include "utilities.h"
#include "RandomDevice.h"
#include "path_planning.h"

using namespace std;


/*-----------------------------------------------------------*/
/*                     Finds nearby IDs                      */
/*-----------------------------------------------------------*/
void find_nearby(Eigen::ArrayXXd population, vector<double> infection_zone, Eigen::ArrayXd &indices, int &infected_number,
	bool traveling_infects = false, string kind = "healthy", Eigen::ArrayXXd infected_previous_step = {});

/*-----------------------------------------------------------*/
/*                    Find new infections                    */
/*-----------------------------------------------------------*/
void infect(Eigen::ArrayXXd &population, Eigen::ArrayXXd &destinations, Configuration Config, int frame,
	RandomDevice *my_rand, bool send_to_location = false, vector<double> location_bounds = {}, 
	int location_no = 1, double location_odds = 1.0, bool test_flag = false);

/*-----------------------------------------------------------*/
/*                     Recover or die                        */
/*-----------------------------------------------------------*/
void recover_or_die(Eigen::ArrayXXd &population, int frame, Configuration Config, RandomDevice *my_rand);

/*-----------------------------------------------------------*/
/*                     Compute mortality                     */
/*-----------------------------------------------------------*/
void compute_mortality(int age, double &mortality_chance, int risk_age = 50,
	int critical_age = 80, double critical_mortality_chance = 0.5,
	string risk_increase = "linear");

/*-----------------------------------------------------------*/
/*              healthcare population infection              */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd healthcare_infection_correction(Eigen::ArrayXXd worker_population, RandomDevice *my_rand, double healthcare_risk_factor = 0.2);