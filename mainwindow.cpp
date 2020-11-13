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
#include <QDebug>
#include <QDesktopWidget>
#include <QScreen>
#include <QMessageBox>
#include <QMetaEnum>

MainWindow::MainWindow(Configuration *Config_init, QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  
	Config = Config_init;

	ui->setupUi(this);

	// Connect data signal to updator slot
	connect(this, SIGNAL(arrivedsignal(QVector<double>, QVector<double>, 
									   QVector<double>, QVector<double>, 
									   QVector<double>, QVector<double>, 
									   QVector<double>, QVector<double>,
									   int)),
			this, SLOT(realtimeDataInputSlot(QVector<double>, QVector<double>,
											 QVector<double>, QVector<double>,
											 QVector<double>, QVector<double>,
											 QVector<double>, QVector<double>,
											 int)));

	IC_0 = 0.3; // initialize slider to current IC value
	IC_max = 0.5; // maximum slider position
	IC_min = 0.1; // minimum slider position
	SD_0 = 0.1; // initialize slider to current SD value
	SD_max = 0.3; // maximum slider position
	SD_min = 0.0; // minimum slider position
	TC_0 = 10; // initialize slider to current TC value
	TC_max = 40; // maximum slider position
	TC_min = 0; // minimum slider position
	setupDemo(0, Config);
  
	// for making screenshots of the current demo or all demos (for website screenshots):
	//QTimer::singleShot(1500, this, SLOT(allScreenShots()));
	//QTimer::singleShot(4000, this, SLOT(screenShot()));
}

void MainWindow::setupDemo(int demoIndex, Configuration *Config)
{
	switch (demoIndex)
	{
		case 0: setupRealtimeScatterDemo(ui->customPlot, Config); break;
	}
	setWindowTitle("QCustomPlot: "+ demoName);
	statusBar()->clearMessage();
	setGeometry(200, 210, 1100, 500);
	currentDemoIndex = demoIndex;
	ui->customPlot->replot();
}

void MainWindow::setupRealtimeScatterDemo(QCustomPlot *customPlot, Configuration *Config)
{
	//customPlot->setParent(this);
	//customPlot->setBackgroundRole(QPalette::Shadow);
	//customPlot->setStyleSheet("color:black;");

	// configure axis rect:
	customPlot->plotLayout()->clear(); // clear default axis rect so we can start from scratch
	QCPAxisRect *ABMAxisRect = new QCPAxisRect(customPlot); // ABM axis object
	QCPAxisRect *SIRAxisRect = new QCPAxisRect(customPlot); // SIR axis object
	
	//QCPLayoutGrid *subLayout = new QCPLayoutGrid;
	customPlot->plotLayout()->addElement(1, 0, ABMAxisRect); // insert axis rect in first column
	customPlot->plotLayout()->addElement(1, 1, SIRAxisRect); // insert axis rect in second column
	
	// get color palettes
	vector<string> palette = Config->get_palette();

	// Define colors from palette
	QColor S_color, I_color, R_color, F_color;
	S_color.setNamedColor(palette[0].c_str()); // susceptible color
	I_color.setNamedColor(palette[1].c_str()); // infected color
	R_color.setNamedColor(palette[2].c_str()); // recovered color
	F_color.setNamedColor(palette[3].c_str()); // fatalities color

	//=============================================================================//
	// Set up ABM plot first

	demoName = "Pandemic simulation";

	// setup an extra legend for that axis rect:
	QCPLegend *ABMLegend = new QCPLegend;
	ABMAxisRect->insetLayout()->addElement(ABMLegend, Qt::AlignTop | Qt::AlignRight);
	ABMLegend->setLayer("legend");
	ABMLegend->setVisible(true);
	ABMLegend->setFont(QFont("Helvetica", 9));
	ABMLegend->setRowSpacing(-3);
	ABMLegend->setFillOrder(QCPLegend::foColumnsFirst); // make legend horizontal
	customPlot->plotLayout()->addElement(0, 0, ABMLegend);

	customPlot->setAutoAddPlottableToLegend(false); // would add to the main legend (in the primary axis rect)

	// create a graph in the new axis rect:
	QCPGraph *s_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // susceptible dots
	QCPGraph *i_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // infected dots
	QCPGraph *r_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // recovered dots
	QCPGraph *f_points = customPlot->addGraph(ABMAxisRect->axis(QCPAxis::atBottom), ABMAxisRect->axis(QCPAxis::atLeft)); // fatalities dots

	// add a legend item to the new legend, representing the graph:
	ABMLegend->addItem(new QCPPlottableLegendItem(ABMLegend, s_points));
	s_points->setPen(QPen(S_color));
	s_points->setLineStyle(QCPGraph::lsNone);
	s_points->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
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

	// set blank axis lines:
	customPlot->rescaleAxes();
	ABMAxisRect->axis(QCPAxis::atBottom, 0)->setTicks(false); // xAxis
	ABMAxisRect->axis(QCPAxis::atLeft, 0)->setTicks(false); // yAxis
	ABMAxisRect->axis(QCPAxis::atBottom, 0)->setTickLabels(false); // xAxis
	ABMAxisRect->axis(QCPAxis::atLeft, 0)->setTickLabels(false); // yAxis

	if (!Config->self_isolate) {
		ABMAxisRect->setMinimumSize(480, 450); // make ABM axis rect size fixed
		ABMAxisRect->axis(QCPAxis::atBottom, 0)->setRange(Config->xbounds[0] - 0.02, Config->xbounds[1] + 0.02); // xAxis
		ABMAxisRect->axis(QCPAxis::atLeft, 0)->setRange(Config->ybounds[0] - 0.02, Config->ybounds[1] + 0.02); // yAxis
	}
	else if (Config->self_isolate) {
		ABMAxisRect->setMinimumSize(600, 450); // make ABM axis rect size fixed
		ABMAxisRect->axis(QCPAxis::atBottom, 0)->setRange(Config->isolation_bounds[0] - 0.02, Config->xbounds[1] + 0.02); // xAxis
		ABMAxisRect->axis(QCPAxis::atLeft, 0)->setRange(Config->ybounds[0] - 0.02, Config->ybounds[1] + 0.02); // yAxis
	}

	// draw a rectangle
	QCPItemRect* rect = new QCPItemRect(customPlot);
	rect->setPen(QPen(Qt::black));

	QPointF topLeft_coor = QPointF(Config->xbounds[0], Config->ybounds[1]);
	QPointF bottomRight_coor = QPointF(Config->xbounds[1], Config->ybounds[0]);

	rect->topLeft->setCoords(topLeft_coor);
	rect->bottomRight->setCoords(bottomRight_coor);

	if (Config->self_isolate) {
		// draw hospital rectangle
		QCPItemRect* rect_host = new QCPItemRect(customPlot);
		rect_host->setPen(QPen(Qt::black));

		QPointF topLeft_coor_h = QPointF(Config->isolation_bounds[0], Config->isolation_bounds[3]);
		QPointF bottomRight_coor_h = QPointF(Config->isolation_bounds[2], Config->isolation_bounds[1]);

		rect_host->topLeft->setCoords(topLeft_coor_h);
		rect_host->bottomRight->setCoords(bottomRight_coor_h);
	}

	//=============================================================================//
	// Set up SIR plot second

	// setup an extra legend for that axis rect:
	QCPLegend *SIRLegend = new QCPLegend;
	SIRAxisRect->insetLayout()->addElement(SIRLegend, Qt::AlignTop | Qt::AlignRight);
	SIRLegend->setLayer("legend");
	SIRLegend->setVisible(true);
	SIRLegend->setFont(QFont("Helvetica", 9));
	SIRLegend->setRowSpacing(-3);
	SIRLegend->setFillOrder(QCPLegend::foColumnsFirst); // make legend horizontal
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
	//SIRAxisRect->axis(QCPAxis::atBottom, 0)->setRange(Config->xbounds[0] - 0.02, Config->xbounds[1] + 0.02); // xAxis
	SIRAxisRect->axis(QCPAxis::atLeft, 0)->setRange(0, Config->pop_size); // yAxis

	// make left and bottom axes always transfer their ranges to right and top axes:
	//connect(SIRAxisRect->axis(QCPAxis::atBottom, 0), SIGNAL(rangeChanged(QCPRange)), SIRAxisRect->axis(QCPAxis::atTop, 0), SLOT(setRange(QCPRange)));
	//connect(SIRAxisRect->axis(QCPAxis::atLeft, 0), SIGNAL(rangeChanged(QCPRange)), SIRAxisRect->axis(QCPAxis::atRight, 0), SLOT(setRange(QCPRange)));

	//=============================================================================//
	// Set up VARIABLE sliders

	// create empty space for sliders to sit on
	QCPLayoutGrid *bottom_bar = new QCPLayoutGrid;
	customPlot->plotLayout()->addElement(2, 0, bottom_bar);
	bottom_bar->setMinimumSize(300, 70);

	QGridLayout *bottom_bar_layout = new QGridLayout;

	// Essential workers slider
	QLabel *ic_label = new QLabel(this);
	ic_label->setText("Infection chance");
	ic_label->setGeometry(340, 530, 300, 10);

	QSlider *ICslider = new QSlider(Qt::Horizontal, customPlot);
	ICslider->setGeometry(10, 520, 300, 10);
	ICslider->setValue(int(((IC_0 - IC_min) / (IC_max - IC_min)) * 100));
	ICslider->setSliderPosition(int(((IC_0 - IC_min) / (IC_max - IC_min)) * 100));

	connect(ICslider, SIGNAL(valueChanged(int)), this, SLOT(setICValue(int)));

	// social distancing slider
	QLabel *sd_label = new QLabel(this);
	sd_label->setText("Social distancing factor");
	sd_label->setGeometry(340, 560, 300, 10);

	QSlider *SDslider = new QSlider(Qt::Horizontal, customPlot);
	SDslider->setGeometry(10, 550, 300, 10);
	SDslider->setValue(int(((SD_0 - SD_min) / (SD_max - SD_min)) * 100));
	SDslider->setSliderPosition(int(((SD_0 - SD_min) / (SD_max - SD_min)) * 100));

	connect(SDslider, SIGNAL(valueChanged(int)), this, SLOT(setSDValue(int)));

	// Number of tests slider
	QLabel *tc_label = new QLabel(this);
	tc_label->setText("Number of tests per day");
	tc_label->setGeometry(340, 590, 300, 10);

	QSlider *TCslider = new QSlider(Qt::Horizontal, customPlot);
	TCslider->setGeometry(10, 580, 300, 10);
	TCslider->setValue(int(((TC_0 - TC_min) / (TC_max - TC_min)) * 100));
	TCslider->setSliderPosition(int(((TC_0 - TC_min) / (TC_max - TC_min)) * 100));

	connect(TCslider, SIGNAL(valueChanged(int)), this, SLOT(setTCValue(int)));

}

void MainWindow::realtimeDataInputSlot(QVector<double> x0, QVector<double> y0,
									   QVector<double> x1, QVector<double> y1,
									   QVector<double> x2, QVector<double> y2,
									   QVector<double> x3, QVector<double> y3,
									   int frame)
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
		
		// add data to lines:
		ui->customPlot->axisRects()[0]->graphs()[0]->addData(x0, y0);
		ui->customPlot->axisRects()[0]->graphs()[1]->addData(x1, y1);
		ui->customPlot->axisRects()[0]->graphs()[2]->addData(x2, y2);
		ui->customPlot->axisRects()[0]->graphs()[3]->addData(x3, y3);

		//=============================================================================//
		// Operate on SIR plot next

		ui->customPlot->axisRects()[1]->graphs()[0]->addData(frame, x0.size() + x1.size());
		ui->customPlot->axisRects()[1]->graphs()[1]->addData(frame, x1.size());
		ui->customPlot->axisRects()[1]->graphs()[2]->addData(frame, x0.size() + x1.size() + x2.size());
		ui->customPlot->axisRects()[1]->graphs()[3]->addData(frame, x0.size() + x1.size() + x2.size() + x3.size());

		lastPointKey = key;
	}

	ui->customPlot->replot();
	// make axis range scroll with the data (at a constant range size of 8):
	ui->customPlot->axisRects()[1]->axis(QCPAxis::atBottom, 0)->setRange(0, frame+1);

}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::setICValue(int IC_new)
{
	IC_0 = ((IC_max - IC_min) * (IC_new / 99.0)) + IC_min;
}

void MainWindow::setSDValue(int SD_new)
{
	SD_0 = ((SD_max - SD_min) * (SD_new/99.0)) + SD_min;
}

void MainWindow::setTCValue(int TC_new)
{
	TC_0 = int(((TC_max - TC_min) * (TC_new / 99.0)) + TC_min);
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
	QString fileName = demoName.toLower()+".png";
	fileName.replace(" ", "");
	pm.save("./screenshots/"+fileName);
	qApp->quit();
}

void MainWindow::allScreenShots()
{
#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
	QPixmap pm = QPixmap::grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#elif QT_VERSION < QT_VERSION_CHECK(5, 5, 0)
	QPixmap pm = qApp->primaryScreen()->grabWindow(qApp->desktop()->winId(), this->x()+2, this->y()+2, this->frameGeometry().width()-4, this->frameGeometry().height()-4);
#else
	QPixmap pm = qApp->primaryScreen()->grabWindow(qApp->desktop()->winId(), this->x()-7, this->y()-7, this->frameGeometry().width()+14, this->frameGeometry().height()+14);
#endif
	QString fileName = demoName.toLower()+".png";
	fileName.replace(" ", "");
	pm.save("./screenshots/"+fileName);
  
	if (currentDemoIndex < 19) {
		if (dataTimer.isActive()) {
			dataTimer.stop();
			dataTimer.disconnect();
			delete ui->customPlot;
			ui->customPlot = new QCustomPlot(ui->centralWidget);
			ui->verticalLayout->addWidget(ui->customPlot);
			setupDemo(currentDemoIndex + 1, Config);
			// setup delay for demos that need time to develop proper look:
			int delay = 250;
			if (currentDemoIndex == 10) // Next is Realtime data demo
				delay = 12000;
			else if (currentDemoIndex == 15) // Next is Item demo
				delay = 5000;
			QTimer::singleShot(delay, this, SLOT(allScreenShots()));
		}
	} else {
		qApp->quit();
	}
}





































