#include "ServoInitiate.h"

std::list<Servo> ServoInitiate::Initiate()
{
    std::list<Servo> servos;
    auto servo = std::make_shared<Servo>("01", 1);
    servos.push_back(*servo);
    return servos;
}