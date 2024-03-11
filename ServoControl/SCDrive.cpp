#include "SCDrive.h"


void ServoOperate::SetServoPosition(int id, int position, int time)
{
    PyObject *pModule = PyImport_ImportModule("ServoManager");
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
                    PyObject *pRetvalue = PyObject_CallMethod(instance, "set_servo_position", "iii", id, position, time);
                    //PyObject *pRetvalue = PyObject_CallMethod(instance, "set_servo_position", "");
                    //if (pRetvalue)
                    //{
                    //    int id;
                    //    PyArg_Parse(pRetvalue, "i", &id);
                    //    const char *idChar = std::to_string(id).c_str();
                    //    ui->ServoNo_Lable->setText(idChar);
                    //}

                }
            }
        }
    }
   
}
    
