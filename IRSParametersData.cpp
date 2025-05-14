#include "include/IRSParametersData.h"

std::string SERVE_DRIVE_LISTENER = "";

std::string URDF_PUBLISHER = "";

std::string URDF_XML = "";

std::string SRDF_XML = "";

std::string MOTION_SERVO_MANAGER = "";

std::string IRS_GROUP_NAME = "";

int TIMEINTERVAL = 0;

double DEGREE_TO_ROTATE_PARAMETER = 0;

double ANGLE_TO_RADIAN = 0;

double SERVO_LINE_TO_ANGLE = 0;

double SERVO_MAX_ANGLEVELOCITY = 0;

double SERVO_MAX_POSITION = 0;

double SERVO_MIN_POSITION = 0;

double SERVO_INITIAL_POSITION = 0;

void InitiateManager::Initial()
{
    SERVE_DRIVE_LISTENER = "serve_drive_listener";
    URDF_PUBLISHER = "urdf_publisher";
    MOTION_SERVO_MANAGER = "motion_servo_manager";
    IRS_GROUP_NAME = "r_arm";

    DEGREE_TO_ROTATE_PARAMETER = 4.16666666666666;//1000.0 / 240;

    SERVO_LINE_TO_ANGLE = 8.18511154838762; //(1.0 / 7/*rervo r = 7mm*/) * (180.0 / 3.14159265358979323846);

    TIMEINTERVAL = 100;

    ANGLE_TO_RADIAN = 0.0174533; //3.14159265358979323846 / 180.0; 

    SERVO_MAX_ANGLEVELOCITY = 0.00748;// degree/ms

    SERVO_MAX_POSITION = 240;

    SERVO_MIN_POSITION = 0;

    SERVO_INITIAL_POSITION = 120;

}
