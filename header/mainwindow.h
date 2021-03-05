/***************************************************************************
**                                                                        **
**  QCustomPlot, an easy to use, modern plotting widget for Qt            **
**  Copyright (C) 2011-2018 Emanuel Eichhammer                            **
**                                                                        **
**  This program is free software: you can redistribute it and/or modify  **
**  it under the terms of the GNU General Public License as published by  **
**  the Free Software Foundation, either version 3 of the License, or     **
**  (at your option) any later version.                                   **
**                                                                        **
**  This program is distributed in the hope that it will be useful,       **
**  but WITHOUT ANY WARRANTY; without even the implied warranty of        **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         **
**  GNU General Public License for more details.                          **
**                                                                        **
**  You should have received a copy of the GNU General Public License     **
**  along with this program.  If not, see http://www.gnu.org/licenses/.   **
**                                                                        **
****************************************************************************
**           Author: Emanuel Eichhammer                                   **
**  Website/Contact: http://www.qcustomplot.com/                          **
**             Date: 25.06.18                                             **
**          Version: 2.0.1                                                **
****************************************************************************/

/************************************************************************************************************
**                                                                                                         **
**  This is the example code for QCustomPlot.                                                              **
**                                                                                                         **
**  It demonstrates basic and some advanced capabilities of the widget. The interesting code is inside     **
**  the "setup(...)Demo" functions of MainWindow.                                                          **
**                                                                                                         **
**  In order to see a demo in action, call the respective "setup(...)Demo" function inside the             **
**  MainWindow constructor. Alternatively you may call setupDemo(i) where i is the index of the demo       **
**  you want (for those, see MainWindow constructor comments). All other functions here are merely a       **
**  way to easily create screenshots of all demos for the website. I.e. a timer is set to successively     **
**  setup all the demos and make a screenshot of the window area and save it in the ./screenshots          **
**  directory.                                                                                             **
**                                                                                                         **
*************************************************************************************************************/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "simulation.h"
#include "utilities.h"

#include <QMainWindow>
#include <QTimer>
#include <QThread>

#include "qcustomplot.h" // the header file of QCustomPlot. Don't forget to add it to your project, if you use an IDE, so it gets compiled.

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT
		QThread workerThread;
public:
	explicit MainWindow(COVID_SIM::simulation *sim_init, QWidget *parent = 0);
	~MainWindow();

	COVID_SIM::simulation *sim;

	double IC_0, SD_0;
	int TC_0;

	int frame_count;

	void setupDemo(int demoIndex);
	void setupRealtimeScatterDemo(QCustomPlot *customPlot);
	void pdfrender();
	void pngrender();
  
public slots:
	void realtimeDataInputSlot(QVector<double> x0, QVector<double> y0,
		QVector<double> x1, QVector<double> y1,
		QVector<double> x2, QVector<double> y2,
		QVector<double> x3, QVector<double> y3,
		QVector<double> x4, QVector<double> y4,
		int frame, float R0, float computation_time,
		QVector<double> x_lower, QVector<double> y_lower,
		QVector<double> x_upper, QVector<double> y_upper);

	void screenShot();
	void iterate_time();

signals:
	void launch_next_step();

private:
	Ui::MainWindow *ui;
	QString demoName;
	QTimer dataTimer;
	QCPItemTracer *itemDemoPhaseTracer;
	int currentDemoIndex;

private slots:
	void on_run_button_clicked();
};

#endif // MAINWINDOW_H
