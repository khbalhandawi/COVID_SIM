#
#  QCustomPlot Plot Examples
#

QT       += core gui sql
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = cpp_corona_simulation
TEMPLATE = app

HEADERS += header/visualizer.h \
    header/utilities.h \
    header/tic_toc.h \
    header/simulation.h \
    header/RandomDevice.h \
    header/qcustomplot.h \
    header/Population_trackers.h \
    header/pch.h \
    header/path_planning.h \
    header/motion.h \
    header/mainwindow.h \
    header/infection.h \
    header/Convert.h \
    header/Configuration.h
	
SOURCES += src/visualizer.cpp \
    src/utilities.cpp \
    src/simulation.cpp \
    src/qcustomplot.cpp \
    src/Population_trackers.cpp \
    src/pch.cpp \
    src/path_planning.cpp \
    src/mainwindow.cpp \
    src/main.cpp \
    src/infection.cpp \
    src/Convert.cpp \
    src/Configuration.cpp \
    src/RandomDevice.cpp \
    src/motion.cpp
	
FORMS += mainwindow.ui

win32: INCLUDEPATH += header \
    Include \
DEPENDPATH += header \
