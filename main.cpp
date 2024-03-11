#include <iostream>
#include <python3.6/Python.h>
#include "mainwindow.h"
#include <QApplication>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    rclcpp::init(argc, argv);

    MainWindow w;
    w.show();

    return a.exec();
} 
