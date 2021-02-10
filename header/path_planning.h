#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include "motion.h"

using namespace std;

/*-----------------------------------------------------------*/
/*                  Send patient to location                 */
/*-----------------------------------------------------------*/
void go_to_location(Eigen::VectorXi Choices, Eigen::ArrayXXf &patients, Eigen::ArrayXXf &destinations, int dest_no = 1);

/*-----------------------------------------------------------*/
/*                Set population destination                 */
/*-----------------------------------------------------------*/
void set_destination(Eigen::ArrayXXf &population, Eigen::ArrayXXf destinations, double travel_speed = 2);

/*-----------------------------------------------------------*/
/*                Check who is at destination                */
/*-----------------------------------------------------------*/
void check_at_destination(Eigen::ArrayXXf &population, Eigen::ArrayXXf destinations, double wander_factor = 1);

/*-----------------------------------------------------------*/
/*            Keeps arrivals within wander range             */
/*-----------------------------------------------------------*/
void keep_at_destination(Eigen::ArrayXXf &population, 
	vector<double> lb_environments, vector<double> ub_environments,
	double wall_buffer, double bounce_buffer);

/*-----------------------------------------------------------*/
/*                 Clear destination markers                 */
/*-----------------------------------------------------------*/
void reset_destinations(Eigen::ArrayXXf &population, vector<int> Ids = {});