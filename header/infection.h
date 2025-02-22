#pragma once

#include "Configuration.h"
#include "RandomDevice.h"

#include<string>
#include<Eigen/Core>

#ifndef INFECTION_H_H
#define INFECTION_H_H

namespace COVID_SIM {

	/*-----------------------------------------------------------*/
	/*                     Finds nearby IDs                      */
	/*-----------------------------------------------------------*/
	void find_nearby(Eigen::ArrayXXf population, Eigen::ArrayXf person_center, double infection_range,
		Eigen::ArrayXf &indices, int &infected_number, bool traveling_infects = false, std::string kind = "healthy",
		std::string shape = "radial", Eigen::ArrayXXf infected_previous_step = {});

	/*-----------------------------------------------------------*/
	/*                      Test and isolate                     */
	/*-----------------------------------------------------------*/
	void test_isolate(Eigen::ArrayXXf &population, Configuration Config, int frame, RandomDevice *my_rand,
		Eigen::ArrayXXf &destinations, int location_no = 1);

	/*-----------------------------------------------------------*/
	/*           Find new infections (within radius)             */
	/*-----------------------------------------------------------*/
	void infect(Eigen::ArrayXXf &population, Eigen::ArrayXXf &destinations, Configuration Config, int frame,
		RandomDevice *my_rand, bool send_to_location = false, int location_no = 1,
		bool test_flag = false, Eigen::ArrayXXf dist = {});

	/*-----------------------------------------------------------*/
	/*                     Recover or die                        */
	/*-----------------------------------------------------------*/
	void recover_or_die(Eigen::ArrayXXf &population, Eigen::ArrayXXf &destinations,
		Configuration Config, int frame, RandomDevice *my_rand, int location_no);

	/*-----------------------------------------------------------*/
	/*                     Compute mortality                     */
	/*-----------------------------------------------------------*/
	void compute_mortality(int age, double &mortality_chance, int risk_age = 50,
		int critical_age = 80, double critical_mortality_chance = 0.5,
		std::string risk_increase = "linear");

	/*-----------------------------------------------------------*/
	/*              healthcare population infection              */
	/*-----------------------------------------------------------*/
	void healthcare_infection_correction(Eigen::ArrayXXf worker_population, RandomDevice *my_rand, double healthcare_risk_factor = 0.2);

}

#endif // INFECTION_H_H