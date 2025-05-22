#ifndef SERVOMANAGER_H
#define SERVOMANAGER_H

#include <tinyxml2.h>
#include <pugixml.hpp>

#include "SCDrive.h"
#include "ServoID.h"

#include "../include/IRSFunction.h"
#include "../include/IRSParametersData.h"

enum ServoDriveMode
{
    None,
    Linear,
    Rotate
};

struct JointParam 
{
    std::string name;
    double wide;
    double high;
    double fix_x;
    double fix_y;
    double r;
};

class ServoManager
{
public:
    ServoManager(std::string jointName, std::shared_ptr<Servo> servo);

    std::string joint_name;

    bool SetServoPosition(double servoRotateAngle, double time);

    bool CheckServoPosition(double beforeJointPosition, double afterJointPosition);

    bool IsUpdateJointPosBound(double& resMax, double& resMin, double max, double min);

    double GetServoRotateAngle(double beforeJointPosition, double afterJointPosition);

    void Initiate(double wide, double high, double fix_x, double fix_y);

    void Initiate(JointParam param);

    void Initiate(double radius);

    void Reset();

private:
    ServoDriveMode drive_mode;

    bool is_initiate = false;

    double servo_position_angle = 0; //angle range is 0~240 degree, servo control value is 0~1000 ; initial angle is 120 degree, initial servo control value is 500. 

    double wide_ = 0;//the length of the negative X-axis when rotating counterclockwise in the plane.

    double high_ = 0;//the length of the positive Y-axis when rotated counterclockwise in the plane.

    double fix_x_ = 0;//the x-coordinate of a fixed point in the plane connected to the rotation point.

    double fix_y_ = 0;//the y-coordinate of a fixed point in the plane connected to the rotation point.

    double r_ = 0;

    std::shared_ptr<Servo> servo_;

    //in xy coordinate system , the top left corner of the initial position is (-a, h) , point p is a fixed point, and the line passes through point p 
    //angle is the angle from the starting
    double CalLineDistance(double wide, double high, double fix_x, double fix_y, double positionAngel);
};

class ServoTools
{
public:
    static std::map<std::string, std::shared_ptr<ServoManager>> Initiate();

    static std::vector<bool> SetServoPositions(std::vector<std::shared_ptr<ServoManager>> servoManagers, std::vector<double> beforeJointPosition, std::vector<double> afterJointPosition, double time);

    static bool SetServoPosition(std::shared_ptr<ServoManager> servoManager, double beforeJointPosition, double afterJointPosition, double time);
    
    static void ResetServo(std::vector<std::shared_ptr<ServoManager>> servoManagers);
};

#endif