#include "SCDrive.h"

ServoOperate::ServoOperate(std::string name, int id)
{
    this->name = name;
    this->id = id;
    pModule = PyImport_ImportModule("ServoManager");
    if (pModule)
    {
        class_obj = PyObject_GetAttrString(pModule, "ServoController");
        if (class_obj)
        {
            PyObject *pArgs = PyTuple_Pack(2, Py_BuildValue("s", "/dev/ttyTHS1"), Py_BuildValue("i", 115200));
            instance = PyObject_CallObject(class_obj, pArgs);
            if(instance)
            {
                isAvaiable = true;
            }
        }
    }
}

ServoOperate::~ServoOperate()
{
    //Py_DECREF(instance);
    //Py_DECREF(class_obj);
    //Py_DECREF(pModule);
}

void ServoOperate::SetServoPosition(int position, double time)
{
    if(isAvaiable)
    PyObject_CallMethod(instance, "set_servo_position", "iii", id, position, time);
    else
    {
        IRS_MESSAGE("servo" + this->name + "error , could not set position");
    }
}

void ServoOperate::SetServoStop()
{
    if(isAvaiable)
    PyObject_CallMethod(instance, "stop", "i", id);
    else
    {
        IRS_MESSAGE("servo" + this->name + "error , could not stop");
    }
} 



Servo::Servo(std::string name, int id)
{
    this->name = name;
    this->id = id;
    this->operate = std::make_unique<ServoOperate>(name, id);
}

void DriveHandle::SetServoPosition(std::list<std::shared_ptr<ServoDriveInfo>> servoInfo)
{
    std::list<std::future<void>> results;
    /*for(auto it = servoInfo.begin(); it != servoInfo.end(); ++it)
    {
        results.push_back(std::async(std::launch::async, [&it](){
            it->servo->operate->SetServoPosition(it->position, it->time);
        }));
    }

    for(auto& resule : results)
    {
        resule.wait();
    }*/

    for(auto it = servoInfo.begin(); it != servoInfo.end(); ++it)
    {       
        it->get()->servo->operate->SetServoPosition(it->get()->position, it->get()->time);      
    }

}
