#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->SetServoNo_Button, SIGNAL(clicked()), this, SLOT(StartROSServoDriveNode()));
}

MainWindow::~MainWindow()
{
    rclcpp::shutdown();
    Py_Finalize();
    delete ui;
}

void MainWindow::Initiate()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./ServoControl')");      
    servos = ServoInitiate::Initiate();
}

void MainWindow::StartROSServoDriveNode()
{
    std::thread([this]()
    {
        servoDriveNodeListener = std::make_shared<ServoDriveNodeListener>(servos);
        rclcpp::spin(servoDriveNodeListener);
    }).detach();
    
}
