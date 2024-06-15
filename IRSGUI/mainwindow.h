#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <python3.6/Python.h>//it must be placed before <QMainWindow>, otherwise an error will be reported
#include <QMainWindow>
#include <rclcpp/rclcpp.hpp>
#include <QScrollBar>
#include <QSharedMemory>
#include <QMessageBox>

#include "../ROSManager/NodeManager.h"
#include "../include/IRSFunction.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
 
public:
    static MainWindow* GetInstance();
    Ui::MainWindow* GetUI();

private:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    MainWindow(const MainWindow&) = delete;
    MainWindow& operator = (const MainWindow&);

    Ui::MainWindow *ui;

    bool isInitROSNode = false;

    std::list<std::shared_ptr<Servo>> servos;

    std::shared_ptr<ServoDriveNodeListenerNode> servoDriveNodeListenerNode;

    void Initiate();

    void Message(std::string message);

    void StartROSServoDriveNode();

private slots:

    void SetServoNo();

    void GetServoNo();

    void ROSNodeInitiate();

    void ServosInitiate();

};

#endif // MAINWINDOW_H
