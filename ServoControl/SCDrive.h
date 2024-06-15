#include <python3.6/Python.h>
#include <memory>
#include <list>
#include <thread>
#include <future>
#include <map>

#include "../include/IRSFunction.h"

class ServoOperate
{
  
public:

    ServoOperate(std::string name, int id);
    ~ServoOperate();
    void SetServoPosition(int position, double time);
    void SetServoStop();
    std::string name = "";
    int id = 0;

private:
    bool isAvaiable = false;
    PyObject *pModule;
    PyObject *class_obj;
    PyObject *instance;
};

class Servo
{
public:
    Servo(std::string name, int id);
    std::unique_ptr<ServoOperate> operate;
    std::string name = "";
    int id = 0;
};

struct ServoDriveInfo
{
    std::shared_ptr<Servo>  servo;
    int position;
    double time;

   ServoDriveInfo(std::shared_ptr<Servo> servo, int position, double time) : servo(servo), position(position), time(time) { }
};

class DriveHandle
{
public:
    static void SetServoPosition(std::list<std::shared_ptr<ServoDriveInfo>> servoInfo);
};

