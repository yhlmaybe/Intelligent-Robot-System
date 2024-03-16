#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <python3.6/Python.h>//it must be placed before <QMainWindow>, otherwise an error will be reported
#include <QMainWindow>
#include <rclcpp/rclcpp.hpp>

#include "../ROSManager/NodeManager.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    std::list<std::shared_ptr<Servo>> servos;

    std::shared_ptr<ServoDriveNodeListener> servoDriveNodeListener;

    void Initiate();

private slots:
    void StartROSServoDriveNode();

};

#endif // MAINWINDOW_H
