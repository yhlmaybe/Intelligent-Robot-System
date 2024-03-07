#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <python3.6/Python.h>//it must be placed before <QMainWindow>, otherwise an error will be reported
#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
    PyObject *pModule;
    PyObject *pDict;
    PyObject *pClassCalc;
    PyObject *pConstruct;
    PyObject *pArgs;
    PyObject *pInstance;
    PyObject *pRetvalue;


private:
    Ui::MainWindow *ui;


private slots:

    void GetServoId();

};

#endif // MAINWINDOW_H
