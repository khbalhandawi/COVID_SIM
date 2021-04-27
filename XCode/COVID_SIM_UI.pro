#
#  QCustomPlot Plot Examples
#

QT       += core gui sql
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = COVID_SIM_UI
TEMPLATE = app

HEADERS += ../header/qcustomplot.h \
    ../header/mainwindow.h \
    ../header/worker.h
	
SOURCES += ../src/qcustomplot.cpp \
    ../src/mainwindow.cpp \
    ../src/worker.cpp \
    ../src/covidsim_ui.cpp
	
FORMS += ../visual_studio/mainwindow.ui

macx: LIBS += -L$$PWD/../build/Release/ -lCOVID_SIM

INCLUDEPATH += $$PWD/../build/Release \
    ../header \
    ../Include
DEPENDPATH += $$PWD/../build/Release \
    ../header

DISTFILES += \
    ../install_tool_commands.bash
