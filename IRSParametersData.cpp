#include "include/IRSParametersData.h"

std::string SERVE_DRIVE_LISTENER = "";

std::string URDF_PUBLISHER = "";

std::string URDF_XML = "";

std::string SRDF_XML = "";

std::string COMPONENT_ROTATE_STATE_PUBLISHER = "";

int TIMEINTERVAL = 0;

double DEGREE_TO_ROTATE_PARAMETER = 0;

void InitiateManager::Initial()
{
    SERVE_DRIVE_LISTENER = "serve_drive_listener";
    URDF_PUBLISHER = "urdf_publisher";
    COMPONENT_ROTATE_STATE_PUBLISHER = "component_rotate_state_publisher";

    DEGREE_TO_ROTATE_PARAMETER = 1000 / 240;
    TIMEINTERVAL = 100;
}
