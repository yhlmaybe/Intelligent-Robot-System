#include <iostream>
#include <python3.6/Python.h>
#include "mainwindow.h"
#include <QApplication>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "include/ROSParametersData.h"

class InitiateManager
{
public:
    static void Initial()
    {
        SERVE_DRIVE_NODE_LISTENER = "serve_drive_node_listener";
        URDF_PUBLISHER = "urdf_publisher";
    }
};


int main(int argc, char *argv[])
{
    InitiateManager::Initial();
    QApplication a(argc, argv);
    rclcpp::init(argc, argv);

    MainWindow *w = MainWindow::GetInstance();
    w->show();

    return a.exec();
}


