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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "Worker.h"

#include <QDebug>
#include <QDesktopWidget>
#include <QScreen>
#include <QMessageBox>
#include <QMetaEnum>

#include <vector>
#include <string>

MainWindow::MainWindow(COVID_SIM::simulation *sim_init, QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{

	sim = sim_init;
	ui->setupUi(this);

	Worker *worker = new Worker(sim);
	worker->moveToThread(&workerThread);
		
	connect(&workerThread, &QThread::finished, worker, &QObject::deleteLater);
	connect(this, &MainWindow::launch_next_step, worker, &Worker::doWork);
	connect(ui->ICslider, SIGNAL(valueChanged(int)), worker, SLOT(setICValue(int)));
	connect(ui->SDslider, SIGNAL(valueChanged(int)), worker, SLOT(setSDValue(int)));
	connect(ui->TCslider, SIGNAL(valueChanged(int)), worker, SLOT(setTCValue(int)));

	connect(worker, &Worker::resultReady, this, &MainWindow::realtimeDataInputSlot);
	connect(worker, &Worker::time_step_finished, this, &MainWindow::iterate_time);
	connect(ui->run_button, &QPushButton::clicked, this, &MainWindow::iterate_time); // invoke first time step slot

	workerThread.start();

	IC_0 = sim->Config.infection_chance; // initialize slider to current IC value
	SD_0 = sim->Config.social_distance_factor / (1e-6 * sim->Config.force_scaling); // initialize slider to current SD value
	TC_0 = sim->Config.number_of_tests; // initialize slider to current TC value

	frame_count = 0;
	setupDemo(0);


	// for making screenshots of the current demo or all demos (for website screenshots):
	//QTimer::singleShot(1500, this, SLOT(allScreenShots()));
	//QTimer::singleShot(4000, this, SLOT(screenShot()));
}

void MainWindow::setupDemo(int demoIndex)
{
	switch (demoIndex)
	{
		case 0: setupRealtimeScatterDemo(ui->customPlot); break;
	}
	setWindowTitle(demoName);
	setWindowIcon(QIcon("covid.png"));
	statusBar()->clearMessage();
	setGeometry(200, 210, 1100, 500);
	currentDemoIndex = demoIndex;
	ui->customPlot->replot();

	if ((sim->Config.save_plot) && (sim->Config.n_plots == 1)) {
		if (sim->Config.self_isolate)  ui->customPlot->setMaximumSize(650, 500); 
		else ui->customPlot->setMaximumSize(580, 500);
	}

}

void MainWindow::setupRealtimeScatterDemo(QCustomPlot *customPlot)
{

	// configure axis rect:
	customPlot->plotLayout()->clear(); // clear default axis rect so we can start from scratch
	customPlot->setBaseSize(1300, 500);

	// background settings
	QColor bg_color;
	bg_color.setNamedColor("#ffffff"); // background color
	customPlot->setBackground(bg_color);

	QRadialGradient gradient(50, 50, 50, 50, 50);
	gradient.setColorAt(0, QColor::fromRgbF(0, 1, 0, 1));
	gradient.setColorAt(1, QColor::fromRgbF(0, 0, 0, 0));

	QBrush brush(gradient);

	//QCPLayoutGrid *subLayout = new QCPLayoutGrid;

	// set axis margins
	customPlot->plotLayout()->setMargins(QMargins(5, 5, 5, 5));

	// get color palettes
	std::vector<std::string> palette = sim->Config.get_palette();

	// Define colors from palette
	QColor S_color, I_color, R_color, F_color, T_color;
	S_color.setNamedColor(palette[0].c_str()); // susceptible color
	I_color.setNamedColor(palette[1].c_str()); // infected color
	R_color.setNamedColor(palette[2].c_str()); // recovered color
	F_color.setNamedColor(palette[3].c_str()); // fatalities color
	T_color.setNamedColor(palette[4].c_str()); // fatalities color

	demoName = "Pandemic simulation";
	//=============================================================================//
	// Set up ABM plot first

	QCPAxisRect *ABMAxisRect = new QCPAxisRect(customPlot); // ABM axis object
	customPlot->plotLayout()->addElement(1, 0, ABMAxisRect); // insert axis rect in first column

	// setup an extra legend for that axis rect:
	QSize legend_dim; legend_dim.setHeight(10); // legend maximum height

	QCPLegend *ABMLegend = new QCPLegend;
	ABMAxisRect->insetLayout()->addElement(ABMLegend, Qt::AlignTop | Qt::AlignRight);
	ABMLegend->setLayer("legend");
	ABMLegend->setVisible(true);

	if (sim->Config.save_plot) ABMLegend->setFont(QFont("Century Gothic", 10)); 
	else ABMLegend->setFont(QFont("Century Gothic", 14));

	ABMLegend->setRowSpacing(-3);
	ABMLegend->setFillOrder(QCPLegend::foColumnsFirst); // make legend horizontal
	ABMLegend->setBorderPen(QPen(Qt::black));
	ABMLegend->setTextColor(Qt::black);
	ABMLegend->setBrush(Qt::NoBrush);
	//ABMLegend->setMaximumSize(legend_dim);

	customPlot->plotLayout()->addElement(0, 0, ABMLegend);

	customPlot->setAutoAddPlottableToLegend(false); // would add to the main legend (in the primary axis rect)

	// create a graph in the new axis rect:
	QCPGraph *s_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // susceptible dots
	QCPGraph *i_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // infected dots
	QCPGraph *r_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // recovered dots
	QCPGraph *f_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // fatalities dots
	QCPGraph *t_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // tracked dots;

	// add a legend item to the new legend, representing the graph:
	ABMLegend->addItem(new QCPPlottableLegendItem(ABMLegend, s_points));
	s_points->setPen(QPen(S_color));
	s_points->setLineStyle(QCPGraph::lsNone);
	s_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
	//s_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QPen(Qt::white, 0.5), S_color, 4));

	s_points->setName("susceptible");

	ABMLegend->addItem(new QCPPlottableLegendItem(ABMLegend, i_points)); // infected dots
	i_points->setPen(QPen(I_color));
	i_points->setLineStyle(QCPGraph::lsNone);
	i_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
	i_points->setName("infected");

	ABMLegend->addItem(new QCPPlottableLegendItem(ABMLegend, r_points)); // recovered dots
	r_points->setPen(QPen(R_color));
	r_points->setLineStyle(QCPGraph::lsNone);
	r_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
	r_points->setName("recovered");

	ABMLegend->addItem(new QCPPlottableLegendItem(ABMLegend, f_points)); // fatalities dots
	f_points->setPen(QPen(F_color));
	f_points->setLineStyle(QCPGraph::lsNone);
	f_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
	f_points->setName("fatalities");

	if ((sim->Config.trace_path) && (sim->Config.track_GC)) {
		ABMLegend->addItem(new QCPPlottableLegendItem(ABMLegend, t_points)); // fatalities dots
		t_points->setPen(QPen(T_color));
		t_points->setLineStyle(QCPGraph::lsNone);
		t_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 8));
		t_points->setName("tracked");
	}

	// set blank axis lines:
	customPlot->rescaleAxes();
	ABMAxisRect->axis(QCPAxis::atBottom, 0)->setTicks(false); // xAxis
	ABMAxisRect->axis(QCPAxis::atLeft, 0)->setTicks(false); // yAxis
	ABMAxisRect->axis(QCPAxis::atBottom, 0)->setTickLabels(false); // xAxis
	ABMAxisRect->axis(QCPAxis::atLeft, 0)->setTickLabels(false); // yAxis
	ABMAxisRect->axis(QCPAxis::atBottom, 0)->setVisible(false); // xAxis
	ABMAxisRect->axis(QCPAxis::atLeft, 0)->setVisible(false); // yAxis
	ABMAxisRect->axis(QCPAxis::atBottom, 0)->grid()->setVisible(false); // xAxis
	ABMAxisRect->axis(QCPAxis::atLeft, 0)->grid()->setVisible(false); // yAxis

	if (!sim->Config.self_isolate) {
		ABMAxisRect->setMinimumSize(480, 450); // make ABM axis rect size fixed
		ABMAxisRect->axis(QCPAxis::atBottom, 0)->setRange(sim->Config.xbounds[0] - 0.02, sim->Config.xbounds[1] + 0.02); // xAxis
		ABMAxisRect->axis(QCPAxis::atLeft, 0)->setRange(sim->Config.ybounds[0] - 0.02, sim->Config.ybounds[1] + 0.02); // yAxis
	}
	else if (sim->Config.self_isolate) {
		ABMAxisRect->setMinimumSize(600, 450); // make ABM axis rect size fixed
		ABMAxisRect->axis(QCPAxis::atBottom, 0)->setRange(sim->Config.isolation_bounds[0] - 0.02, sim->Config.xbounds[1] + 0.02); // xAxis
		ABMAxisRect->axis(QCPAxis::atLeft, 0)->setRange(sim->Config.ybounds[0] - 0.02, sim->Config.ybounds[1] + 0.02); // yAxis
	}

	// draw a rectangle
	QCPItemRect* rect = new QCPItemRect(customPlot);
	rect->setPen(QPen(Qt::black));

	QPointF topLeft_coor = QPointF(sim->Config.xbounds[0], sim->Config.ybounds[1]);
	QPointF bottomRight_coor = QPointF(sim->Config.xbounds[1], sim->Config.ybounds[0]);

	rect->topLeft->setCoords(topLeft_coor);
	rect->bottomRight->setCoords(bottomRight_coor);

	if (sim->Config.self_isolate) {
		// draw hospital rectangle
		QCPItemRect* rect_host = new QCPItemRect(customPlot);
		rect_host->setPen(QPen(Qt::black));

		QPointF topLeft_coor_h = QPointF(sim->Config.isolation_bounds[0], sim->Config.isolation_bounds[3]);
		QPointF bottomRight_coor_h = QPointF(sim->Config.isolation_bounds[2], sim->Config.isolation_bounds[1]);

		rect_host->topLeft->setCoords(topLeft_coor_h);
		rect_host->bottomRight->setCoords(bottomRight_coor_h);
	}

	//=============================================================================//
	// Set up SIR plot second
	if (sim->Config.n_plots == 2) {

		QCPAxisRect *SIRAxisRect = new QCPAxisRect(customPlot); // SIR axis object
		customPlot->plotLayout()->addElement(1, 1, SIRAxisRect); // insert axis rect in second column

		// setup an extra legend for that axis rect:
		QCPLegend *SIRLegend = new QCPLegend;
		SIRAxisRect->insetLayout()->addElement(SIRLegend, Qt::AlignTop | Qt::AlignRight);
		SIRLegend->setLayer("legend");
		SIRLegend->setVisible(true);
		SIRLegend->setFont(QFont("Century Gothic", 14));
		SIRLegend->setRowSpacing(-3);
		SIRLegend->setFillOrder(QCPLegend::foColumnsFirst); // make legend horizontal
		SIRLegend->setBorderPen(QPen(Qt::black));
		SIRLegend->setTextColor(Qt::black);
		SIRLegend->setBrush(Qt::NoBrush);

		//SIRLegend->setMaximumSize(legend_dim);

		customPlot->plotLayout()->addElement(0, 1, SIRLegend);

		customPlot->setAutoAddPlottableToLegend(false); // would add to the main legend (in the primary axis rect)

		// create a graph in the new axis rect:
		QCPGraph *s_graph = customPlot->addGraph(SIRAxisRect->axis(QCPAxis::atBottom), SIRAxisRect->axis(QCPAxis::atLeft)); // susceptible graph
		QCPGraph *i_graph = customPlot->addGraph(SIRAxisRect->axis(QCPAxis::atBottom), SIRAxisRect->axis(QCPAxis::atLeft)); // infected graph
		QCPGraph *r_graph = customPlot->addGraph(SIRAxisRect->axis(QCPAxis::atBottom), SIRAxisRect->axis(QCPAxis::atLeft)); // recovered graph
		QCPGraph *f_graph = customPlot->addGraph(SIRAxisRect->axis(QCPAxis::atBottom), SIRAxisRect->axis(QCPAxis::atLeft)); // fatalities graph

		// add a legend item to the new legend, representing the graph:
		SIRLegend->addItem(new QCPPlottableLegendItem(SIRLegend, s_graph));
		s_graph->setPen(QPen(S_color));
		//s_graph->setLineStyle(QCPGraph::lsNone);
		s_graph->setName("susceptible");
		s_graph->setBrush(QBrush(S_color)); // first graph will be filled with translucent blue

		SIRLegend->addItem(new QCPPlottableLegendItem(SIRLegend, i_graph)); // infected graph
		i_graph->setPen(QPen(I_color));
		i_graph->setName("infected");
		i_graph->setBrush(QBrush(I_color)); // second graph will be filled with translucent red

		SIRLegend->addItem(new QCPPlottableLegendItem(SIRLegend, r_graph)); // recovered graph
		r_graph->setPen(QPen(R_color));
		r_graph->setName("recovered");
		r_graph->setBrush(QBrush(R_color)); // third graph will be filled with translucent grey

		SIRLegend->addItem(new QCPPlottableLegendItem(SIRLegend, f_graph)); // fatalities graph
		f_graph->setPen(QPen(F_color));
		f_graph->setName("fatalities");
		f_graph->setBrush(QBrush(F_color)); // fourth graph will be filled with translucent black

		s_graph->setChannelFillGraph(i_graph); // fill between S and I graphs
		r_graph->setChannelFillGraph(s_graph); // fill between R and S graphs
		f_graph->setChannelFillGraph(r_graph); // fill between F and R graphs

		SIRAxisRect->setMinimumSize(480, 450); // make ABM axis rect size fixed
		//SIRAxisRect->axis(QCPAxis::atBottom, 0)->setRange(sim->Config.xbounds[0] - 0.02, sim->Config.xbounds[1] + 0.02); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setRange(0, sim->Config.pop_size); // yAxis

		// Axis labels
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->setLabel("Simulation steps"); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setLabel("Population size"); // yAxis
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->setLabelFont(QFont("Century Gothic", 14)); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setLabelFont(QFont("Century Gothic", 14)); // yAxis

		// change axis colors
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->setSubTickPen(QPen(Qt::black)); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setSubTickPen(QPen(Qt::black)); // yAxis
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->setTickPen(QPen(Qt::black)); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setTickPen(QPen(Qt::black)); // yAxis
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->setBasePen(QPen(Qt::black)); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setBasePen(QPen(Qt::black)); // yAxis
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->setTickLabelColor(Qt::black); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->setTickLabelColor(Qt::black); // yAxis

		// turn off grid lines
		SIRAxisRect->axis(QCPAxis::atBottom, 0)->grid()->setVisible(false); // xAxis
		SIRAxisRect->axis(QCPAxis::atLeft, 0)->grid()->setVisible(false); // yAxis
	}

	// make left and bottom axes always transfer their ranges to right and top axes:
	//connect(SIRAxisRect->axis(QCPAxis::atBottom, 0), SIGNAL(rangeChanged(QCPRange)), SIRAxisRect->axis(QCPAxis::atTop, 0), SLOT(setRange(QCPRange)));
	//connect(SIRAxisRect->axis(QCPAxis::atLeft, 0), SIGNAL(rangeChanged(QCPRange)), SIRAxisRect->axis(QCPAxis::atRight, 0), SLOT(setRange(QCPRange)));

	//=============================================================================//
	// Set up VARIABLE sliders

	// Essential workers slider
	ui->ICslider->setGeometry(10, 520, 300, 10);
	ui->ICslider->setValue(int(((IC_0 - sim->Config.IC_min) / (sim->Config.IC_max - sim->Config.IC_min)) * 100));
	ui->ICslider->setSliderPosition(int(((IC_0 - sim->Config.IC_min) / (sim->Config.IC_max - sim->Config.IC_min)) * 100));

	// social distancing slider
	ui->SDslider->setGeometry(10, 550, 300, 10);
	ui->SDslider->setValue(int(((SD_0 - sim->Config.SD_min) / (sim->Config.SD_max - sim->Config.SD_min)) * 100));
	ui->SDslider->setSliderPosition(int(((SD_0 - sim->Config.SD_min) / (sim->Config.SD_max - sim->Config.SD_min)) * 100));

	// Number of tests slider
	ui->TCslider->setGeometry(10, 580, 300, 10);
	ui->TCslider->setValue(int(((TC_0 - sim->Config.TC_min) / (sim->Config.TC_max - sim->Config.TC_min)) * 100));
	ui->TCslider->setSliderPosition(int(((TC_0 - sim->Config.TC_min) / (sim->Config.TC_max - sim->Config.TC_min)) * 100));


	//=============================================================================//
	// Set up R0 display
	//ui->R0_box->setMinimumSize(200, 50);

}

void MainWindow::realtimeDataInputSlot(QVector<double> x0, QVector<double> y0,
									   QVector<double> x1, QVector<double> y1,
									   QVector<double> x2, QVector<double> y2,
									   QVector<double> x3, QVector<double> y3,
									   QVector<double> x4, QVector<double> y4,
									   int frame, float R0, float computation_time,
									   QVector<double> x_lower, QVector<double> y_lower, 
									   QVector<double> x_upper, QVector<double> y_upper)
{
	static QTime time(QTime::currentTime());
	// calculate two new data points:
	double key = time.elapsed() / 1000.0; // time elapsed since start of demo, in seconds
	static double lastPointKey = 0;
	if (key - lastPointKey > 0.002) // at most add point every 2 ms
	{
		//=============================================================================//
		// Operate on ABM plot first

		// clear old data:
		ui->customPlot->axisRects()[0]->graphs()[0]->data()->clear();
		ui->customPlot->axisRects()[0]->graphs()[1]->data()->clear();
		ui->customPlot->axisRects()[0]->graphs()[2]->data()->clear();
		ui->customPlot->axisRects()[0]->graphs()[3]->data()->clear();
		if ((sim->Config.trace_path) && (sim->Config.track_GC)) ui->customPlot->axisRects()[0]->graphs()[4]->data()->clear();
		
		// add data to lines:
		ui->customPlot->axisRects()[0]->graphs()[0]->addData(x0, y0);
		ui->customPlot->axisRects()[0]->graphs()[1]->addData(x1, y1);
		ui->customPlot->axisRects()[0]->graphs()[2]->addData(x2, y2);
		ui->customPlot->axisRects()[0]->graphs()[3]->addData(x3, y3);
		if ((sim->Config.trace_path) && (sim->Config.track_GC)) ui->customPlot->axisRects()[0]->graphs()[4]->addData(x4, y4);

		//=============================================================================//
		// Operate on SIR plot next
		if (sim->Config.n_plots == 2) {
			ui->customPlot->axisRects()[1]->graphs()[0]->addData(frame, x0.size() + x1.size());
			ui->customPlot->axisRects()[1]->graphs()[1]->addData(frame, x1.size());
			ui->customPlot->axisRects()[1]->graphs()[2]->addData(frame, x0.size() + x1.size() + x2.size());
			ui->customPlot->axisRects()[1]->graphs()[3]->addData(frame, x0.size() + x1.size() + x2.size() + x3.size());
		}

		//=============================================================================//
		// Display R0 value
		QString R0_q;
		R0_q.sprintf("%.2f", R0);
		ui->R0_indicator->setText(R0_q);

		lastPointKey = key;
	}

	if ((sim->Config.trace_path) && (sim->Config.track_GC)) {

		for (int i = 0; i < x_lower.size(); i++) {
			// draw a rectangle around an individual
			QCPItemRect* rect_trace = new QCPItemRect(ui->customPlot);
			rect_trace->setPen(QPen(Qt::red));

			QPointF topLeft_coor = QPointF(x_lower[i], y_upper[i]);
			QPointF bottomRight_coor = QPointF(x_upper[i], y_lower[i]);

			rect_trace->topLeft->setCoords(topLeft_coor);
			rect_trace->bottomRight->setCoords(bottomRight_coor);
		}

	}

	ui->customPlot->replot();
	if (sim->Config.n_plots == 2) {
		// make axis range scroll with the data (at a constant range size of 8):
		ui->customPlot->axisRects()[1]->axis(QCPAxis::atBottom, 0)->setRange(0, frame + 1);
	}

	// calculate frames per second:
	static double lastFpsKey;
	static int frameCount;
	++frameCount;
	if (key - lastFpsKey > 0.2) // average fps over 0.2 seconds
	{
		ui->statusBar->showMessage(
			QString("%1 FPS, Total Data points: %2")
			.arg(frameCount / (key - lastFpsKey), 0, 'f', 0)
			.arg(frame), 0);
		lastFpsKey = key;
		frameCount = 0;
	}

	frame_count = frame;

	// take a screenshot
	if ((sim->Config.save_plot) && ((frame % sim->Config.save_pop_freq) == 0)) {
		//QTimer::singleShot(4000, this, SLOT(screenShot()));
		pdfrender(); // only works in debug mode
	}

	int sleep_time = (1000 / 60) - computation_time - (time.elapsed() - key); // target frame rate = 60 FPS
	if (sleep_time > 0) {
		// Block the calling thread for x milliseconds
		QThread::msleep(sleep_time);
	}

}

void MainWindow::on_run_button_clicked()
{
	bool run_action = ui->run_button->isChecked();

	if (run_action) {
		ui->run_button->setText("run");
	}
	else {
		ui->run_button->setText("pause");
	}
}

void MainWindow::screenShot()
{
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
	QPixmap pm = QPixmap::grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#elif QT_VERSION < QT_VERSION_CHECK(5, 5, 0)
	QPixmap pm = qApp->primaryScreen()->grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#else
	QPixmap pm = qApp->primaryScreen()->grabWindow(qApp->desktop()->winId(), this->x()-7, this->y()-7, this->frameGeometry().width()+14, this->frameGeometry().height()+14);
#endif
	QString fileName = demoName.toLower() + "_" + QString::number(frame_count) + ".png";
	fileName.replace(" ", "");
	pm.save("./screenshots/"+fileName);
}

void MainWindow::iterate_time()
{
	if (ui->run_button->isChecked()) {
		emit launch_next_step();
	}
}

void MainWindow::pdfrender()
{
	QString folder = QString::fromStdString(sim->Config.plot_path);
	QString fileName = "./" + folder + "/sim_" + QString::number(frame_count) + ".pdf";
	ui->customPlot->savePdf(fileName);
}

MainWindow::~MainWindow()
{
	workerThread.quit();
	workerThread.wait();
	delete ui;
}