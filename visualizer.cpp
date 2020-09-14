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
 \file   visualizer.cpp
 \brief  Contains all methods for visualization tasks (implementation)
 \author Khalil Al Handawi
 \date   2010-08-25
 \see    visualizer.h
 */

#include "visualizer.h"

 /*-----------------------------------------------------------*/
 /*                       Constructor                         */
 /*-----------------------------------------------------------*/
visualizer::visualizer()
{
}

/*-----------------------------------------------------------*/
/*                       Build figure                        */
/*-----------------------------------------------------------*/
void visualizer::build_fig(Configuration Config, vector<int> fig_size)
{
	plt::backend("WXAgg"); //https://github.com/lava/matplotlib-cpp/issues/95
	if (!Config.self_isolate) {
		plt::figure_size(1000, 500);
		plt::xlim(Config.xbounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}
	else if (Config.self_isolate) {
		plt::figure_size(1200, 500);
		plt::xlim(Config.isolation_bounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}

	plt::subplot(1, 2, 1);
	// plt.title('infection simulation')
	plt::xlim(Config.xbounds[0], Config.xbounds[1]);
	plt::ylim(Config.ybounds[0], Config.ybounds[1]);

	vector<double> lower_corner = { Config.xbounds[0], Config.ybounds[0] };
	double width = Config.xbounds[1] - Config.xbounds[0];
	double height = Config.ybounds[1] - Config.ybounds[0];

	// Draw boundary of world
	string bound_color = "k";
	Rectangle(lower_corner, width, height, "1.0", bound_color);

	if (Config.self_isolate) {
		build_hospital(Config.isolation_bounds[0], Config.isolation_bounds[2],
					   Config.isolation_bounds[1], Config.isolation_bounds[3], bound_color, Config.add_cross);
	}

	plt::axis("off");

	// SIR graph
	plt::subplot(1, 2, 2);
	plt::ylim(0, Config.pop_size);

	plt::xlabel("Simulation Steps");
	plt::ylabel("Number of people");

	map<string, string> keywords_legend;
	keywords_legend["loc"] = "upper center";
	keywords_legend["ncol"] = "5";
	keywords_legend["fontsize"] = "10";

	plt::legend();

	if (Config.save_plot) {
		check_folder(Config.plot_path); // create save directory
	}

}

/*-----------------------------------------------------------*/
/*             Build figure (scatter plot only)              */
/*-----------------------------------------------------------*/
void visualizer::build_fig_scatter(Configuration Config, vector<int> fig_size)
{
	plt::backend("WXAgg"); //https://github.com/lava/matplotlib-cpp/issues/95
	if (!Config.self_isolate) {
		plt::figure_size(500, 500);
		// plt.title('infection simulation')
		plt::xlim(Config.xbounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);

	}
	else if (Config.self_isolate) {
		plt::figure_size(700, 500);
		plt::xlim(Config.isolation_bounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}

	vector<double> lower_corner = { Config.xbounds[0], Config.ybounds[0] };
	double width = Config.xbounds[1] - Config.xbounds[0];
	double height = Config.ybounds[1] - Config.ybounds[0];

	// Draw boundary of world
	string bound_color = "k";
	Rectangle(lower_corner, width, height, "1.0", bound_color);

	if (Config.self_isolate) {
		build_hospital(Config.isolation_bounds[0], Config.isolation_bounds[2],
			Config.isolation_bounds[1], Config.isolation_bounds[3], bound_color, Config.add_cross);
	}

	//plt::axis("off");

	if (Config.save_plot) {
		check_folder(Config.plot_path); // create save directory
	}

}

/*-----------------------------------------------------------*/
/*                      Build SIR figure                     */
/*-----------------------------------------------------------*/
void visualizer::build_fig_SIR(Configuration Config, vector<int> fig_size)
{
	plt::backend("WXAgg"); //https://github.com/lava/matplotlib-cpp/issues/95
	plt::figure_size(700, 500);

	plt::ylim(0, Config.pop_size + 100);

	plt::xlabel("Simulation Steps");
	plt::ylabel("Number of people");

	if (Config.save_plot) {
		check_folder(Config.plot_path); // create save directory
		cout << Config.plot_path << endl;
	}
}

/*-----------------------------------------------------------*/
/*                  Update figure time step                  */
/*-----------------------------------------------------------*/
void visualizer::draw_tstep(Configuration Config, Eigen::ArrayXXd population, Population_trackers pop_tracker, int frame)
{
	//construct plot and visualise

	// get color palettes
	vector<string> palette = Config.get_palette();
	// Clear first subplot
	plt::subplot(1, 2, 1);
	plt::cla();

	/*--------------------------------------------------*/
	// Replot all world and boundaries
	if (!Config.self_isolate) {
		plt::xlim(Config.xbounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}
	else if (Config.self_isolate) {
		plt::xlim(Config.isolation_bounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}

	vector<double> lower_corner = { Config.xbounds[0], Config.ybounds[0] };
	double width = Config.xbounds[1] - Config.xbounds[0];
	double height = Config.ybounds[1] - Config.ybounds[0];

	// Draw boundary of world
	string bound_color = "k";
	Rectangle(lower_corner, width, height, "1.0", bound_color);

	if (Config.self_isolate) {
		build_hospital(Config.isolation_bounds[0], Config.isolation_bounds[2],
					   Config.isolation_bounds[1], Config.isolation_bounds[3], bound_color, false);
	}

	plt::axis("off");

	/*--------------------------------------------------*/
	// plot population segments

	Eigen::ArrayXXd susceptible = population(select_rows(population.col(6) == 0), { 1,2 });
	Eigen::ArrayXXd infected = population(select_rows(population.col(6) == 1), { 1,2 });
	Eigen::ArrayXXd recovered = population(select_rows(population.col(6) == 2), { 1,2 });
	Eigen::ArrayXXd fatalities = population(select_rows(population.col(6) == 3), { 1,2 });

	vector<double> susceptible_x(susceptible.rows()); Map<ArrayXd>(&susceptible_x[0], susceptible.rows(), 1) = susceptible.col(0);
	vector<double> infected_x(infected.rows()); Map<ArrayXd>(&infected_x[0], infected.rows(), 1) = infected.col(0);
	vector<double> recovered_x(recovered.rows()); Map<ArrayXd>(&recovered_x[0], recovered.rows(), 1) = recovered.col(0);
	vector<double> fatalities_x(fatalities.rows()); Map<ArrayXd>(&fatalities_x[0], fatalities.rows(), 1) = fatalities.col(0);

	vector<double> susceptible_y(susceptible.rows()); Map<ArrayXd>(&susceptible_y[0], susceptible.rows(), 1) = susceptible.col(1);
	vector<double> infected_y(infected.rows()); Map<ArrayXd>(&infected_y[0], infected.rows(), 1) = infected.col(1);
	vector<double> recovered_y(recovered.rows()); Map<ArrayXd>(&recovered_y[0], recovered.rows(), 1) = recovered.col(1);
	vector<double> fatalities_y(fatalities.rows()); Map<ArrayXd>(&fatalities_y[0], fatalities.rows(), 1) = fatalities.col(1);

	map<string, string> keywords;
	keywords["color"] = palette[0];
	plt::scatter(susceptible_x, susceptible_y, Config.marker_size, keywords);
	keywords["color"] = palette[1];
	plt::scatter(infected_x, infected_y, Config.marker_size, keywords);
	keywords["color"] = palette[2];
	plt::scatter(recovered_x, recovered_y, Config.marker_size, keywords);
	keywords["color"] = palette[3];
	plt::scatter(fatalities_x, fatalities_y, Config.marker_size, keywords);

	//add text descriptors
	string output_string = "timestep: " + to_string(frame) +
		" total: " + to_string(population.rows()) +
		" susceptible: " + to_string(susceptible.rows()) +
		" infected: " + to_string(infected.rows()) +
		" recovered: " + to_string(recovered.rows()) +
		" fatalities: " + to_string(fatalities.rows());

	if (!Config.self_isolate) {
		plt::text(Config.xbounds[0], Config.ybounds[1] + ((Config.ybounds[1] - Config.ybounds[0]) / 100), output_string);
	}
	else if (Config.self_isolate) {
		plt::text(Config.isolation_bounds[0], Config.ybounds[1] + ((Config.ybounds[1] - Config.ybounds[0]) / 100), output_string);
	}

	plt::draw();

	/*--------------------------------------------------*/
	// plot sir diagram
	plt::subplot(1, 2, 2);

	if (Config.treatment_dependent_risk) {
		vector<int> infected_arr = pop_tracker.infectious;

		vector<int> line_hc(infected_arr.size());
		int value = Config.healthcare_capacity;
		fill(line_hc.begin(), line_hc.end(), value);

		map<string, string> keywords;
		keywords["color"] = "red";
		keywords["linestyle"] = ":";
		keywords["linewidth"] = "2";
		keywords["label"] = "healthcare capacity";

		plt::plot(line_hc, keywords);
	}


	if (Config.plot_mode == "default") {

		map<string, string> keywords;
		keywords["color"] = palette[1];

		plt::plot(pop_tracker.infectious, keywords);

		keywords["color"] = palette[3];
		keywords["label"] = "fatalities";

		plt::plot(pop_tracker.fatalities, keywords);
	}
	else if (Config.plot_mode == "sir") {

		vector<int> s, i, r, f, rs, rr, rf;

		s = pop_tracker.susceptible;
		i = pop_tracker.infectious;
		r = pop_tracker.recovered;
		f = pop_tracker.fatalities;

		transform(s.begin(), s.end(), i.begin(), s.begin(), plus<int>()); // s + i
		rr = s;
		transform(rr.begin(), rr.end(), r.begin(), rr.begin(), plus<int>()); // s + r
		rf = rr;
		transform(rf.begin(), rf.end(), f.begin(), rf.begin(), plus<int>()); // rr + f

		// ax2.plot(i, color = palette[1], label = 'infectious')
		// ax2.plot(s, color = palette[0], label = 'susceptible')
		// ax2.plot(rr, color = palette[2], label = 'recovered')
		// ax2.plot(rf, color = palette[3], label = 'fatalities')

		// filled plot
		map<string, string> keywords;

		vector<int> x = sequence(0, (frame + 1) );
		vector<int> line(i.size());
		fill(line.begin(), line.end(), 0);

		keywords["color"] = palette[1];
		plt::fill_between(x, line, i, keywords); // infectious
		keywords["color"] = palette[0];
		plt::fill_between(x, i, s, keywords); // susceptible
		keywords["color"] = palette[2];
		plt::fill_between(x, s, rr, keywords); // recovered
		keywords["color"] = palette[3];
		plt::fill_between(x, rr, rf, keywords); // fatalities
	}

	plt::draw();
	plt::pause(0.0001);

	if (Config.save_plot) {

		string bg_color = "w";
		string save_path = Config.plot_path + "/" + to_string(frame) + ".png";
		map<string, string> keywords;
		keywords["dpi"] = "300";
		keywords["facecolor"] = bg_color;

		plt::save(save_path);

	}
}

/*-----------------------------------------------------------*/
/*       Update figure time step (scatter plot only)         */
/*-----------------------------------------------------------*/
void visualizer::draw_tstep_scatter(Configuration Config, Eigen::ArrayXXd population, Population_trackers pop_tracker, int frame)
{
	//construct plot and visualise

	// get color palettes
	vector<string> palette = Config.get_palette();
	// Clear first subplot
	plt::cla();

	/*--------------------------------------------------*/
	// Replot all world and boundaries
	if (!Config.self_isolate) {
		plt::xlim(Config.xbounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}
	else if (Config.self_isolate) {
		plt::xlim(Config.isolation_bounds[0] - 0.02, Config.xbounds[1] + 0.02);
		plt::ylim(Config.ybounds[0] - 0.02, Config.ybounds[1] + 0.02);
	}

	vector<double> lower_corner = { Config.xbounds[0], Config.ybounds[0] };
	double width = Config.xbounds[1] - Config.xbounds[0];
	double height = Config.ybounds[1] - Config.ybounds[0];

	// Draw boundary of world
	string bound_color = "k";
	Rectangle(lower_corner, width, height, "1.0", bound_color);

	if (Config.self_isolate) {
		build_hospital(Config.isolation_bounds[0], Config.isolation_bounds[2],
			Config.isolation_bounds[1], Config.isolation_bounds[3], bound_color, Config.add_cross);
	}

	plt::axis("off");

	/*--------------------------------------------------*/
	// plot population segments
	Eigen::ArrayXXd susceptible = population(select_rows(population.col(6) == 0), { 1,2 });
	Eigen::ArrayXXd infected = population(select_rows(population.col(6) == 1), { 1,2 });
	Eigen::ArrayXXd recovered = population(select_rows(population.col(6) == 2), { 1,2 });
	Eigen::ArrayXXd fatalities = population(select_rows(population.col(6) == 3), { 1,2 });

	vector<double> susceptible_x(susceptible.rows()); Map<ArrayXd>(&susceptible_x[0], susceptible.rows(), 1) = susceptible.col(0);
	vector<double> infected_x(infected.rows()); Map<ArrayXd>(&infected_x[0], infected.rows(), 1) = infected.col(0);
	vector<double> recovered_x(recovered.rows()); Map<ArrayXd>(&recovered_x[0], recovered.rows(), 1) = recovered.col(0);
	vector<double> fatalities_x(fatalities.rows()); Map<ArrayXd>(&fatalities_x[0], fatalities.rows(), 1) = fatalities.col(0);

	vector<double> susceptible_y(susceptible.rows()); Map<ArrayXd>(&susceptible_y[0], susceptible.rows(), 1) = susceptible.col(1);
	vector<double> infected_y(infected.rows()); Map<ArrayXd>(&infected_y[0], infected.rows(), 1) = infected.col(1);
	vector<double> recovered_y(recovered.rows()); Map<ArrayXd>(&recovered_y[0], recovered.rows(), 1) = recovered.col(1);
	vector<double> fatalities_y(fatalities.rows()); Map<ArrayXd>(&fatalities_y[0], fatalities.rows(), 1) = fatalities.col(1);

	map<string, string> keywords;
	keywords["color"] = palette[0];
	plt::scatter(susceptible_x, susceptible_y, Config.marker_size, keywords);
	keywords["color"] = palette[1];
	plt::scatter(infected_x, infected_y, Config.marker_size, keywords);
	keywords["color"] = palette[2];
	plt::scatter(recovered_x, recovered_y, Config.marker_size, keywords);
	keywords["color"] = palette[3];
	plt::scatter(fatalities_x, fatalities_y, Config.marker_size, keywords);

	// Trace path of random individual
	if (Config.trace_path) {
		Eigen::ArrayXXd grid_coords = pop_tracker.grid_coords;
		Eigen::ArrayXd ground_covered = pop_tracker.ground_covered.row(0);
		Eigen::ArrayXXd active_grids = grid_coords(select_rows(ground_covered != 0), Eigen::all);

		Eigen::ArrayXd grid;
		keywords["color"] = "r";
		keywords["marker"] = "s";
		vector<double> corner;
		for (int i = 0; i < active_grids.rows(); i++) {
			grid = active_grids.row(i);
			corner = { grid[0], grid[1] };
			Rectangle(corner, grid[2] - grid[0], grid[3] - grid[1], "1.0", "r");
		}
	}

	//add text descriptors
	string output_string = "timestep: " + to_string(frame) +
		" total: " + to_string(population.rows()) +
		" susceptible: " + to_string(susceptible.rows()) +
		" infected: " + to_string(infected.rows()) +
		" recovered: " + to_string(recovered.rows()) +
		" fatalities: " + to_string(fatalities.rows());

	if (!Config.self_isolate) {
		plt::text(Config.xbounds[0], Config.ybounds[1] + ((Config.ybounds[1] - Config.ybounds[0]) / 100), output_string);
	}
	else if (Config.self_isolate) {
		plt::text(Config.isolation_bounds[0], Config.ybounds[1] + ((Config.ybounds[1] - Config.ybounds[0]) / 100), output_string);
	}

	plt::draw();
	plt::pause(0.001);

	if (Config.save_plot) {

		string bg_color = "w";
		string save_path = Config.plot_path + "/" + to_string(frame) + ".png";
		map<string, string> keywords;
		keywords["dpi"] = "300";
		keywords["facecolor"] = bg_color;

		plt::save(save_path);

	}

}

/*-----------------------------------------------------------*/
/*            Update figure time step (SIR only)             */
/*-----------------------------------------------------------*/
void visualizer::draw_SIRonly(Configuration Config, Eigen::ArrayXXd population, Population_trackers pop_tracker, int frame)
{
	// construct plot and visualise

	// get color palettes
	vector<string> palette = Config.get_palette();

	if (Config.treatment_dependent_risk) {
		vector<int> infected_arr = pop_tracker.infectious;

		vector<int> line_hc(infected_arr.size());
		int value = Config.healthcare_capacity;
		fill(line_hc.begin(), line_hc.end(), value);

		map<string, string> keywords;
		keywords["color"] = "red";
		keywords["linestyle"] = ":";
		keywords["linewidth"] = "2";
		keywords["label"] = "healthcare capacity";

		plt::plot(line_hc, keywords);
	}

	if (Config.plot_mode == "default") {

		map<string, string> keywords;
		keywords["color"] = palette[1];
		keywords["label"] = "infectious";

		plt::plot(pop_tracker.infectious, keywords);

		keywords["color"] = palette[3];
		keywords["label"] = "fatalities";

		plt::plot(pop_tracker.fatalities, keywords);
	}
	else if (Config.plot_mode == "sir") {

		vector<int> s, i, r, f;

		s = pop_tracker.susceptible;
		i = pop_tracker.infectious;
		r = pop_tracker.recovered;
		f = pop_tracker.fatalities;

		// filled plot
		map<string, string> keywords;
		keywords["linewidth"] = "1.5";
		vector<int> x = sequence(0, (frame + 1) );

		keywords["color"] = palette[0];
		keywords["label"] = "susceptible";
		plt::plot(x, s, keywords); // susceptible
		keywords["color"] = palette[1];
		keywords["label"] = "infectious";
		plt::plot(x, i, keywords); // infectious
		keywords["color"] = palette[2];
		keywords["label"] = "recovered";
		plt::plot(x, r, keywords); // recovered
		keywords["color"] = palette[3];
		keywords["label"] = "fatalities";
		plt::plot(x, f, keywords); // fatalities
	}

	plt::legend();

	plt::draw();
	plt::pause(0.001);

	if (Config.save_plot) {

		string bg_color = "w";
		string save_path = Config.plot_path + "\\" + "Final_SIR" + ".pdf";
		map<string, string> keywords;
		//keywords["dpi"] = "300";
		keywords["facecolor"] = bg_color;

		plt::save(save_path);

	}
}

/*-----------------------------------------------------------*/
/*                     Draw a rectangle                      */
/*-----------------------------------------------------------*/
void visualizer::Rectangle(vector<double> lower_corner, double width, double height, string linewidth, string edgecolor)
{
	map<string, string> keywords;
	keywords["color"] = edgecolor;
	keywords["linewidth"] = linewidth;
	
	plt::plot({ lower_corner[0], lower_corner[0] + width }, { lower_corner[1], lower_corner[1] }, keywords);
	plt::plot({ lower_corner[0], lower_corner[0] }, { lower_corner[1], lower_corner[1] + height }, keywords);
	plt::plot({ lower_corner[0], lower_corner[0] + width }, { lower_corner[1] + height , lower_corner[1] + height }, keywords);
	plt::plot({ lower_corner[0] + width, lower_corner[0] + width }, { lower_corner[1], lower_corner[1] + height }, keywords);
}

/*-----------------------------------------------------------*/
/*                      Draw a hospital                      */
/*-----------------------------------------------------------*/
void visualizer::build_hospital(double xmin, double xmax, double ymin, double ymax, string bound_color, bool addcross)
{
	/*builds hospital

	Defines hospital and returns wall coordinates for
	the hospital, as well as coordinates for a red cross
	above it

	Keyword arguments
	-----------------
	xmin : int or float
	lower boundary on the x axis

	xmax : int or float
	upper boundary on the x axis

	ymin : int or float
	lower boundary on the y axis

	ymax : int or float
	upper boundary on the y axis

	plt : matplotlib.pyplot object
	the plot object to which to append the hospital drawing
	if None, coordinates are returned

	Returns
	-------
	None
	*/

	// plot walls
	vector<double> lower_corner = { xmin, ymin };
	double width = xmax - xmin;
	double height = ymax - ymin;

	// Draw boundary of destination
	Rectangle(lower_corner, width, height, "1.0", bound_color);

	// plot red cross
	if (addcross) {
		double xmiddle = xmin + ((xmax - xmin) / 2);
		height = min({ 0.3, (ymax - ymin) / 5 });

		std::map<string, string> keywords;
		keywords["color"] = "red";
		keywords["linewidth"] = "5.0";

		double offset = 0.02;

		plt::plot({ xmiddle, xmiddle }, { ymax + offset, ymax + height + offset }, keywords);
		plt::plot({ xmiddle - (height / 2), xmiddle + (height / 2) }, { ymax + (height / 2) + offset, ymax + (height / 2) + offset }, keywords);
	}


}

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
visualizer::~visualizer()
{
}
