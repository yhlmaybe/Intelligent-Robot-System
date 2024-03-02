#include <iostream>
#include <python3.6/Python.h>
#include <QApplication>
#include "InitiateUI.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    Dialog dialog;
    dialog.show();

    return a.exec();

} 
