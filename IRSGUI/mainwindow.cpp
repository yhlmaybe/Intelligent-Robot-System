#include "mainwindow.h"
#include "ui_mainwindow.h"

void IRS_MESSAGE(std::string message)
{
    MainWindow* mainWindow = MainWindow::GetInstance();
    auto ui = mainWindow->GetUI();

    ui->MessageText->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
    QString QMessage = QString::fromStdString(message);
    ui->MessageText->insertPlainText(QMessage);
    QScrollBar *scrollbar = ui->MessageText->verticalScrollBar();
    if(scrollbar)  
    {
        scrollbar->setSliderPosition(scrollbar->maximum());
    }  
}


MainWindow* MainWindow::GetInstance()
{
    static MainWindow instance;
    return &instance;
}

Ui::MainWindow* MainWindow::GetUI()
{
    return ui;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    if(!QSharedMemory("IRSUniqueKey").create(1))
    {
        QMessageBox::warning(this, "warning", "program has been launched");
        QApplication::exit(0);
        return;
    }
    ui->setupUi(this);

    connect(ui->SetServoNo_Button, SIGNAL(clicked()), this, SLOT(StartROSServoDriveNode()));
}

MainWindow::~MainWindow()
{
    rclcpp::shutdown();
    Py_Finalize();
    delete ui; 
    QSharedMemory sharedMemory("IRSUniqueKey");
    sharedMemory.detach();
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
        servoDriveNodeListenerNode = std::make_shared<ServoDriveNodeListenerNode>(servos);
        rclcpp::spin(servoDriveNodeListenerNode);
    }).detach();

    ROSNodeManager::UrdfInitial();

    
}
