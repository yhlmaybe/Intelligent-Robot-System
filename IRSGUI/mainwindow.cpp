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

    Initiate();

    connect(ui->SetServoNo_Button, SIGNAL(clicked()), this, SLOT(SetServoNo()));
    connect(ui->GetServoNo_Button, SIGNAL(clicked()), this, SLOT(GetServoNo()));
    
}

MainWindow::~MainWindow()
{
    rclcpp::shutdown();
    delete ui; 
    QSharedMemory sharedMemory("IRSUniqueKey");
    sharedMemory.detach();
    Py_Finalize();
}

void MainWindow::Initiate()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./ServoControl')");      
    servos = ServoInitiate::Initiate();
}

void MainWindow::Message(std::string message)
{
    ui->MessageText->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
    QString QMessage = QString::fromStdString(message);
    ui->MessageText->insertPlainText(QMessage);
    QScrollBar *scrollbar = ui->MessageText->verticalScrollBar();
    if(scrollbar)  
    {
        scrollbar->setSliderPosition(scrollbar->maximum());
    }  
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

void MainWindow::SetServoNo()
{
    PyObject *pModule;
    pModule = PyImport_ImportModule("ServoManager");
    if (pModule)
    {

        PyObject *class_obj = PyObject_GetAttrString(pModule, "ServoController");
        if (class_obj)
        {
            PyObject *pArgs = PyTuple_Pack(2, Py_BuildValue("s", "/dev/ttyTHS1"), Py_BuildValue("i", 115200));
            if (pArgs)
            {
                PyObject *instance = PyObject_CallObject(class_obj, pArgs);
                if (instance)
                {
                    PyObject *pRetvalue = PyObject_CallMethod(instance, "get_servo_id", "");
                    if (pRetvalue)
                    {
                        int id;
                        PyArg_Parse(pRetvalue, "i", &id);
                        try
                        {    
                            int newIdInt = std::stoi(ui->ServoNo_LineEdit->text().toStdString());    
                            pRetvalue = PyObject_CallMethod(instance, "set_servo_id", "ii", id, newIdInt);
                        }
                        catch(const std::exception& e)
                        {
                            Message("id error");
                        }
                    }
                }
            }
        }
    }
}

void MainWindow::GetServoNo()
{
    PyObject *pModule;
    pModule = PyImport_ImportModule("ServoManager");
    if (pModule)
    {

        PyObject *class_obj = PyObject_GetAttrString(pModule, "ServoController");
        if (class_obj)
        {
            PyObject *pArgs = PyTuple_Pack(2, Py_BuildValue("s", "/dev/ttyTHS1"), Py_BuildValue("i", 115200));
            if (pArgs)
            {
                PyObject *instance = PyObject_CallObject(class_obj, pArgs);
                if (instance)
                {
                    PyObject *pRetvalue = PyObject_CallMethod(instance, "get_servo_id", "");
                    if (pRetvalue)
                    {
                        int id;
                        PyArg_Parse(pRetvalue, "i", &id);
                        const char *idChar = std::to_string(id).c_str();
                        ui->ServoNo_Lable->setText(idChar);
                    }
                }
            }
        }
    }
}
