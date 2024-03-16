#include <python3.6/Python.h>
#include <memory>
#include <list>
#include <thread>
#include <future>
#include <map>

class ServoOperate
{
  
public:

    ServoOperate(std::string name, int id);
    ~ServoOperate();
    void SetServoPosition(int position, int time);
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


class DriveHandle
{
public:
    static void SetServoPosition(std::map<Servo, int> servoPositions, int time);
};

