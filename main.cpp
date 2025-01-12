#include <iostream>
#include <python3.10/Python.h>
#include "mainwindow.h"
#include <QApplication>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "include/IRSParametersData.h"

int main(int argc, char *argv[])
{
    InitiateManager::Initial();

    rclcpp::init(argc, argv);
    QApplication a(argc, argv);


    MainWindow *w = MainWindow::GetInstance();
    w->show();

    return a.exec();
}


