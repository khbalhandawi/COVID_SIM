#pragma once

#include "utilities.h"
#include "RandomDevice.h"
#include "Configuration.h"
#include "Population_trackers.h"
#include "matplotlibcpp.h"
#include <algorithm>

using namespace std;
namespace plt = matplotlibcpp;

class visualizer
{
public:

	/*-----------------------------------------------------------*/
	/*                       Build figure                        */
	/*-----------------------------------------------------------*/
	void build_fig(Configuration Config, vector<int> fig_size = { 5,10 });

	/*-----------------------------------------------------------*/
	/*             Build figure (scatter plot only)              */
	/*-----------------------------------------------------------*/
	void build_fig_scatter(Configuration Config, vector<int> fig_size = { 5,10 });

	/*-----------------------------------------------------------*/
	/*                  Build figure (SIR only)                  */
	/*-----------------------------------------------------------*/
	void build_fig_SIR(Configuration Config, vector<int> fig_size = { 5,10 });

	/*-----------------------------------------------------------*/
	/*                  Update figure time step                  */
	/*-----------------------------------------------------------*/
	void draw_tstep(Configuration Config, Eigen::ArrayXXd population, Population_trackers pop_tracker, int frame);

	/*-----------------------------------------------------------*/
	/*       Update figure time step (scatter plot only)         */
	/*-----------------------------------------------------------*/
	void draw_tstep_scatter(Configuration Config, Eigen::ArrayXXd population, Population_trackers pop_tracker, int frame);

	/*-----------------------------------------------------------*/
	/*            Update figure time step (SIR only)             */
	/*-----------------------------------------------------------*/
	void draw_SIRonly(Configuration Config, Eigen::ArrayXXd population, Population_trackers pop_tracker, int frame);

	/*-----------------------------------------------------------*/
	/*                     Draw a rectangle                      */
	/*-----------------------------------------------------------*/
	void Rectangle(vector<double> lower_corner, double width, double height, string linewidth = "1.0", string edgecolor = "k");

	/*-----------------------------------------------------------*/
	/*                      Draw a hospital                      */
	/*-----------------------------------------------------------*/
	void build_hospital(double xmin, double xmax, double ymin, double ymax, string bound_color, bool addcross = true);

	/*-----------------------------------------------------------*/
	/*                        Constructor                        */
	/*-----------------------------------------------------------*/
	visualizer();

	/*-----------------------------------------------------------*/
	/*                        Destructor                         */
	/*-----------------------------------------------------------*/
	~visualizer();
};
