#
#  QCustomPlot Plot Examples
#

QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = cpp_corona_simulation
TEMPLATE = app

SOURCES += main.cpp\
			Configuration.cpp\
			Convert.cpp\
			infection.cpp\
			motion.cpp\
			path_planning.cpp\
			Population_trackers.cpp\
			RandomDevice.cpp\
			simulation.cpp\
			tic_toc.cpp\
			utilities.cpp\
			visualizer.cpp\
           mainwindow.cpp \
		   qcustomplot.cpp

HEADERS  += Configuration.h\
			Convert.h\
			infection.h\
			motion.h\
			path_planning.h\
			Population_trackers.h\
			RandomDevice.h\
			simulation.h\
			tic_toc.h\
			utilities.h\
			visualizer.h\
			mainwindow.h \
			qcustomplot.h

FORMS    += mainwindow.ui

