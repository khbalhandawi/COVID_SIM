#pragma once

#include "RandomDevice.h"
#include "Configuration.h"
#include "utilities.h"
#include "motion.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace Eigen;
//#include <EigenRand/EigenRand>

class Population_trackers
{
public:

	vector<int> susceptible;
	vector<int> infectious;
	vector<int> recovered;
	vector<int> fatalities;
	Configuration Config;
	vector<double> distance_travelled;
	Eigen::ArrayXf total_distance; // distance travelled by individuals
	vector<double> mean_perentage_covered;
	vector<double> mean_R0; // mean basic reproductive number
	Eigen::ArrayXXf grid_coords;
	Eigen::ArrayXXf ground_covered;
	Eigen::ArrayXf perentage_covered; // portion of world covered by individuals
	// PLACEHOLDER - whether recovered individual can be reinfected
	bool reinfect;

	/*-----------------------------------------------------------*/
	/*                      Update counts                        */
	/*-----------------------------------------------------------*/
	void update_counts(Eigen::ArrayXXf population, int frame);
	
	/*-----------------------------------------------------------*/
	/*                       Constructor                         */
	/*-----------------------------------------------------------*/
	Population_trackers(Configuration Config, Eigen::ArrayXXf grid_coords, Eigen::ArrayXXf ground_covered);

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~Population_trackers();
};

/*-----------------------------------------------------------*/
/*                  initialize population                    */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf initialize_population(Configuration Config, RandomDevice *my_rand, int mean_age = 45, int max_age = 105,
	vector<double> xbounds = { 0, 1 }, vector<double> ybounds = { 0, 1 });

/*-----------------------------------------------------------*/
/*                 initialize destinations                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXf initialize_destination_matrix(int pop_size, int total_destinations);

/*-----------------------------------------------------------*/
/*                initialize ground covered                  */
/*-----------------------------------------------------------*/
void initialize_ground_covered_matrix(Eigen::ArrayXXf &grid_coords, Eigen::ArrayXXf &ground_covered, int pop_size, int n_gridpoints, vector<double> xbounds = { 0, 1 },
	vector<double> ybounds = { 0, 1 });

/*-----------------------------------------------------------*/
/*                   destination bounds                      */
/*-----------------------------------------------------------*/
void set_destination_bounds(Eigen::ArrayXXf &population, Eigen::ArrayXXf &destinations, double xmin, double ymin,
	double xmax, double ymax, RandomDevice *my_rand, int dest_no = 1, bool teleport = true);

/*-----------------------------------------------------------*/
/*                   save population data                    */
/*-----------------------------------------------------------*/
void save_data(Eigen::ArrayXXf population, Population_trackers pop_tracker, Configuration Config, int frame, string folder);

/*-----------------------------------------------------------*/
/*         save population data at current time step         */
/*-----------------------------------------------------------*/
void save_population(Eigen::ArrayXXf population, int tstep = 0, string folder = "data_tstep");

/*-----------------------------------------------------------*/
/*    save population ground covered at current time step    */
/*-----------------------------------------------------------*/
void save_ground_covered(Eigen::ArrayXXf ground_covered, int tstep = 0, string folder = "data_tstep");

/*-----------------------------------------------------------*/
/*                   save grid coordinates                   */
/*-----------------------------------------------------------*/
void save_grid_coords(Eigen::ArrayXXf grid_coords, string folder = "data_tstep");