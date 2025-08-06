#include <iostream>
#include <python3.8/Python.h>
#include <QApplication>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "include/IRSParametersData.h"
#include "IRSGUI/mainwindow.h"

int main(int argc, char *argv[])
{
    InitiateManager::Initial();

    rclcpp::init(argc, argv);
    QApplication a(argc, argv);


    MainWindow *w = MainWindow::GetInstance();
    w->show();

    return a.exec();
}


