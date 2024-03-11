#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //connect(ui->SetServoNo_Button, SIGNAL(clicked()), this, SLOT(GetServoId()));
}

MainWindow::~MainWindow()
{
    delete ui;
    rclcpp::shutdown();
     Py_Finalize();
}

void MainWindow::Initiate()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./ServoControl')");
       
}
