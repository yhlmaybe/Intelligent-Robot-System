#include "mainwindow.h"
#include "ui_mainwindow.h"

void IRS_MESSAGE(std::string message)
{
    MainWindow *mainWindow = MainWindow::GetInstance();
    QString QMessage = QString::fromStdString(message);
    QMetaObject::invokeMethod(
        mainWindow,
        "SetMessage",
        Qt::AutoConnection,
        Q_ARG(QString, QMessage));
}

void IRS_FORM_MESSAGE(std::string message, MessageFunction fun)
{
    MainWindow *mainWindow = MainWindow::GetInstance();
    QString QMessage = QString::fromStdString(message);
    std::string functionName = "";
    switch (fun)
    {
    case MessageFunction::JointDatas:
        functionName = "SetJointDatasFormDatas";
        break;
    case MessageFunction::EndEffectorDatas:
        functionName = "SetEndEffectorDatasFormDatas";
        break;
    case MessageFunction::BrainDeepLearnFormDatas:
        functionName = "SetBrainDeepLearnFormDatas";
        break;
    default:
        functionName = "SetMessage";
    }
    QMetaObject::invokeMethod(
        mainWindow,
        functionName.c_str(),
        Qt::AutoConnection,
        Q_ARG(QString, QMessage));
}

void IRS_MESSAGE(const char* format, ...) 
{
    va_list args;
    va_start(args, format);
    int size = std::vsnprintf(nullptr, 0, format, args) + 1; 
    va_end(args);
    char* buffer = new char[size];
    va_start(args, format);
    std::vsnprintf(buffer, size, format, args);
    va_end(args);
    std::string result(buffer);
    delete[] buffer;
    IRS_MESSAGE(result);
}

MainWindow* MainWindow::GetInstance()
{
    static MainWindow* instance = new MainWindow();
    return instance;
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

    joint_datas_form = new JointDatasForm();
    end_effector_datas_form = new EndEffectorDatasForm();
    brain_deep_learn_form = new BrainDeepLearnForm();

    forms.push_back(joint_datas_form);
    forms.push_back(end_effector_datas_form);
    forms.push_back(brain_deep_learn_form);

    
    connect(this, &MainWindow::SetJointDatasFormDatas, joint_datas_form, &JointDatasForm::AddData);
    connect(this, &MainWindow::SetEndEffectorDatasFormDatas, end_effector_datas_form, &EndEffectorDatasForm::AddData);
    connect(this, &MainWindow::SetBrainDeepLearnFormDatas, brain_deep_learn_form, &BrainDeepLearnForm::AddData);

    connect(ui->actionJoint_Datas, &QAction::triggered, this, &MainWindow::OpenJointDatasForm);
    connect(ui->actionEnd_Effector_Datas, &QAction::triggered, this, &MainWindow::OpenEndEffectDatasForm);
    connect(ui->actionBrainDeepLearn_Setting, &QAction::triggered, this, &MainWindow::OpenBrainDeepLearnForm);

    connect(ui->SetServoNo_Button, SIGNAL(clicked()), this, SLOT(SetServoNo()));
    connect(ui->GetServoNo_Button, SIGNAL(clicked()), this, SLOT(GetServoNo()));
    connect(ui->Initiate_Button, SIGNAL(clicked()), this, SLOT(Initiate()));
    connect(ui->StateReset_Button, SIGNAL(clicked()), this, SLOT(StateReset()));
    connect(ui->setJointPosition_Button, SIGNAL(clicked()), this, SLOT(SetJointPosition()));
    connect(ui->setGoalPoint_Button, SIGNAL(clicked()), this, SLOT(SetGoalPoint()));
    connect(ui->start_Button, SIGNAL(clicked()), this, SLOT(Start()));
    connect(ui->end_Button, SIGNAL(clicked()), this, SLOT(End()));
}

MainWindow::~MainWindow()
{  
    QSharedMemory sharedMemory("IRSUniqueKey");
    sharedMemory.detach();
    delete ui; 
}

void MainWindow::closeEvent(QCloseEvent *event) 
{
    IRSCoreHandle::End();
    rclcpp::shutdown();
    for(QWidget *form : forms)
    {
        if(form->isVisible())
        {
            form->close();
        }
    }
    event->accept();
}

void MainWindow::Initiate()
{
    if (!is_initial)
    {
        py_manager = std::make_shared<PythonInteraction::Manager>();
        py_manager->SetPrintCallback([this](const char* p, std::size_t n, const std::string& name) {this->SetPythonMessageToTextBrowser(p, int(n), name);});
    }
    is_initial = true;
}

void MainWindow::SetMessage(QString QMessage)
{
    std::lock_guard<std::mutex> lock(message_mtx);
    ui->MessageText->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
    ui->MessageText->insertPlainText("\n");
    ui->MessageText->insertPlainText(QMessage);
    QScrollBar *scrollbar = ui->MessageText->verticalScrollBar();
    if(scrollbar)  
    {
        scrollbar->setSliderPosition(scrollbar->maximum());
    }  
}

void MainWindow::OpenJointDatasForm()
{
    joint_datas_form->show();
}

void MainWindow::OpenEndEffectDatasForm()
{
    end_effector_datas_form->show();
}

void MainWindow::OpenBrainDeepLearnForm()
{
    brain_deep_learn_form->show();
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
            PyObject *pArgs = PyTuple_Pack(2, Py_BuildValue("s", "/dev/ttyTHS0"), Py_BuildValue("i", 115200));
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
                            SetMessage("id error");
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
            PyObject *pArgs = PyTuple_Pack(2, Py_BuildValue("s", "/dev/ttyTHS0"), Py_BuildValue("i", 115200));
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

void MainWindow::Start()
{
    if (!is_running)
    {
        IRSCoreHandle::Start();
        is_running = true;

        ui->jointName_comboBox->clear();
        std::vector<std::string> joint_names = IRSCoreHandle::GetCurrentRobotState()->getVariableNames();
        for (std::string name : joint_names)
        {
            ui->jointName_comboBox->addItem(QString::fromStdString(name));
        }
    }
    else
    {
        SetMessage("already initialized");     
    }
}

void MainWindow::End()
{
    IRSCoreHandle::End();
    is_running = false;
}

void MainWindow::StateReset()
{
    if(is_running)
    {
        IRSCoreHandle::ResetServoState();
    }
    else 
    {
        IRS_MESSAGE("the state needs to be initialized before reset");
    }
}

void MainWindow::SetJointPosition()
{
    if(is_running)
    {
        QString joint_name = ui->jointName_comboBox->currentText().trimmed();
        QString position = ui->jointPosition_lineEdit->text().trimmed();

        if(joint_name.isEmpty() || position.isEmpty()) 
        {
            IRS_MESSAGE("joint name or position is empty");
            return;
        }

        bool ok;
        double position_double = position.toDouble(&ok);
        if(!ok)
        {
            IRS_MESSAGE("invalid position format");
            return;
        }

        IRSCoreHandle::SetJointPosition(joint_name.toStdString(), position_double);
    }
    else 
    {
        IRS_MESSAGE("the joint needs to be initialized before set position");
    }
}

void MainWindow::SetGoalPoint()
{
    if (is_running)
    {
        QString goal_point = ui->goalPoint_lineEdit->text().trimmed();

        if (goal_point.isEmpty())
        {
            IRS_MESSAGE("goal point is empty");
            return;
        }

        QStringList parts = goal_point.split(",", QString::SkipEmptyParts);

        if (parts.size() != 3)
        {
            IRS_MESSAGE("the number of numbers separated by commas is not three");
            return;
        }

        bool ok1, ok2, ok3;
        double x = parts[0].trimmed().toDouble(&ok1);
        double y = parts[1].trimmed().toDouble(&ok2);
        double z = parts[2].trimmed().toDouble(&ok3);

        if (!ok1 || !ok2 || !ok3)
        {
            IRS_MESSAGE("requires input containing non-numeric characters");
            return;
        }

        std::vector<Eigen::Vector3d> point_vec{Eigen::Vector3d(x, y, z)};
        IRSCoreHandle::GetGoalPointsQueue().push(point_vec);
    }
    else
    {
        IRS_MESSAGE("the joint needs to be initialized before set position");
    }
}

void MainWindow::SetPythonMessageToTextBrowser(const char *data, std::size_t len, const std::string &mouduleName)
{
    std::string str(data, len); 
    if(mouduleName == "Manager")
    {
        IRS_FORM_MESSAGE(str, MessageFunction::BrainDeepLearnFormDatas);
    }
}