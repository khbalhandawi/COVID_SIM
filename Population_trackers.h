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
	Eigen::ArrayXd total_distance; // distance travelled by individuals
	vector<double> mean_perentage_covered;
	Eigen::ArrayXXd grid_coords;
	Eigen::ArrayXXd ground_covered;
	Eigen::ArrayXd perentage_covered; // portion of world covered by individuals
	// PLACEHOLDER - whether recovered individual can be reinfected
	bool reinfect;

	/*-----------------------------------------------------------*/
	/*                      Update counts                        */
	/*-----------------------------------------------------------*/
	void update_counts(Eigen::ArrayXXd population, int frame);
	
	/*-----------------------------------------------------------*/
	/*                       Constructor                         */
	/*-----------------------------------------------------------*/
	Population_trackers(Configuration Config, Eigen::ArrayXXd grid_coords, Eigen::ArrayXXd ground_covered);

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~Population_trackers();
};

/*-----------------------------------------------------------*/
/*                  initialize population                    */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd initialize_population(Configuration Config, RandomDevice *my_rand, int mean_age = 45, int max_age = 105,
	vector<double> xbounds = { 0, 1 }, vector<double> ybounds = { 0, 1 });

/*-----------------------------------------------------------*/
/*                 initialize destinations                   */
/*-----------------------------------------------------------*/
Eigen::ArrayXXd initialize_destination_matrix(int pop_size, int total_destinations);

/*-----------------------------------------------------------*/
/*                initialize ground covered                  */
/*-----------------------------------------------------------*/
tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> initialize_ground_covered_matrix(int pop_size, int n_gridpoints, vector<double> xbounds = { 0, 1 },
	vector<double> ybounds = { 0, 1 });

/*-----------------------------------------------------------*/
/*                   destination bounds                      */
/*-----------------------------------------------------------*/
tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> set_destination_bounds(Eigen::ArrayXXd population, Eigen::ArrayXXd destinations, double xmin, double ymin,
	double xmax, double ymax, RandomDevice *my_rand, int dest_no = 1, bool teleport = true);

/*-----------------------------------------------------------*/
/*                   save population data                    */
/*-----------------------------------------------------------*/
void save_data(Eigen::ArrayXXd population, Population_trackers pop_tracker);

/*-----------------------------------------------------------*/
/*         save population data at current time step         */
/*-----------------------------------------------------------*/
void save_population(Eigen::ArrayXXd population, int tstep = 0, string folder = "data_tstep");