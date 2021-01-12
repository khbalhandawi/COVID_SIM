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

#ifndef _N_QT
#include "visualizer.h"
 /*-----------------------------------------------------------*/
 /*                       Constructor                         */
 /*-----------------------------------------------------------*/
visualizer::visualizer()
{
}

 /*-----------------------------------------------------------*/
 /*                    Start a Qt thread                      */
 /*-----------------------------------------------------------*/
std::unique_ptr<MainWindow> visualizer::start_qt(Configuration Config)
{

	int argc = 0;
	char **argv = NULL;

	std::unique_ptr<MainWindow> mainWindow = nullptr; // initialize null pointer to mainwindow

	// Start the Qt realtime plot demo in a worker thread
	std::thread myThread
	(
 		[&] {
 		QApplication application(argc, argv);
 		mainWindow = std::make_unique<MainWindow>(&Config); // lambda capture by reference
 		mainWindow->show();

 		return application.exec();
	}
	);

	qRegisterMetaType<QVector<double> >("QVector<double>"); // register QVector<double> for queued connection type

	return mainWindow;

	}

/*-----------------------------------------------------------*/
/*                     Update Qt window                      */
/*-----------------------------------------------------------*/
void visualizer::update_qt(Eigen::ArrayXXf population,
	int frame, float R0, std::unique_ptr<MainWindow> &mainWindow)
{

	// Initialize Qt Vectors
	QVector<double> susceptible_x, susceptible_y,
		infected_x, infected_y,
		recovered_x, recovered_y,
		fatalities_x, fatalities_y;

	/*--------------------------------------------------*/
	// plot population segments
	Eigen::ArrayXXd susceptible = population(select_rows(population.col(6) == 0), { 1,2 }).cast<double>();
	Eigen::ArrayXXd infected = population(select_rows(population.col(6) == 1), { 1,2 }).cast<double>();
	Eigen::ArrayXXd recovered = population(select_rows(population.col(6) == 2), { 1,2 }).cast<double>();
	Eigen::ArrayXXd fatalities = population(select_rows(population.col(6) == 3), { 1,2 }).cast<double>();

	if (susceptible.rows() > 0) { // to avoid assertion errors
		susceptible_x.resize(susceptible.rows()); Map<ArrayXd>(&susceptible_x[0], susceptible.rows(), 1) = susceptible.col(0);
		susceptible_y.resize(susceptible.rows()); Map<ArrayXd>(&susceptible_y[0], susceptible.rows(), 1) = susceptible.col(1);
	}

	if (infected.rows() > 0) { // to avoid assertion errors
		infected_x.resize(infected.rows()); Map<ArrayXd>(&infected_x[0], infected.rows(), 1) = infected.col(0);
		infected_y.resize(infected.rows()); Map<ArrayXd>(&infected_y[0], infected.rows(), 1) = infected.col(1);
	}

	if (recovered.rows() > 0) { // to avoid assertion errors
		recovered_x.resize(recovered.rows()); Map<ArrayXd>(&recovered_x[0], recovered.rows(), 1) = recovered.col(0);
		recovered_y.resize(recovered.rows()); Map<ArrayXd>(&recovered_y[0], recovered.rows(), 1) = recovered.col(1);
	}

	if (fatalities.rows() > 0) { // to avoid assertion errors
		fatalities_x.resize(fatalities.rows()); Map<ArrayXd>(&fatalities_x[0], fatalities.rows(), 1) = fatalities.col(0);
		fatalities_y.resize(fatalities.rows()); Map<ArrayXd>(&fatalities_y[0], fatalities.rows(), 1) = fatalities.col(1);
	}

	// Block the calling thread for x milliseconds // http://www.cplusplus.com/reference/thread/this_thread/sleep_for/
	std::this_thread::sleep_for(std::chrono::milliseconds(1));

	// update mainwindow using new data
	emit mainWindow->arrivedsignal(susceptible_x, susceptible_y,
		infected_x, infected_y,
		recovered_x, recovered_y,
		fatalities_x, fatalities_y,
		frame, R0); // emit signal

}

/*-----------------------------------------------------------*/
/*                        Destructor                         */
/*-----------------------------------------------------------*/
visualizer::~visualizer()
{
}

#endif