#ifndef IRSPARAMETERSDATE_H
#define IRSPARAMETERSDATE_H

#include <string>

extern double DEGREE_TO_ROTATE_PARAMETER;

extern int TIMEINTERVAL;

extern double ANGLE_TO_RADIAN;

extern double SERVO_LINE_TO_ANGLE;

extern double SERVO_MAX_ANGLEVELOCITY;

extern std::string SERVE_DRIVE_LISTENER;

extern std::string URDF_PUBLISHER;

extern std::string MOTION_SERVO_MANAGER;

extern std::string URDF_XML;

extern std::string SRDF_XML;

extern std::string IRS_GROUP_NAME;

extern double SERVO_MAX_POSITION;

extern double SERVO_MIN_POSITION;

extern double SERVO_INITIAL_POSITION;


class InitiateManager
{
public:
    static void Initial();
};

#endif