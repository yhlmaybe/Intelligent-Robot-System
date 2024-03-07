#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);



    connect(ui->SetServoNo_Button, SIGNAL(clicked()), this, SLOT(GetServoId()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::GetServoId()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./ServoControl')");
    pModule = PyImport_ImportModule("ServoManager");
    if (pModule)
    {

        PyObject *class_obj = PyObject_GetAttrString(pModule, "ServoController");
        if (class_obj)
        {
            pArgs = PyTuple_Pack(2, Py_BuildValue("s", "/dev/ttyTHS1"), Py_BuildValue("i", 115200));
            if (pArgs)
            {
                PyObject *instance = PyObject_CallObject(class_obj, pArgs);
                if (instance)
                {
                    pRetvalue = PyObject_CallMethod(instance, "get_servo_id", "");
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
    Py_Finalize();
}
