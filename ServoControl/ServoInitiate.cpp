#include "ServoInitiate.h"

std::list<std::shared_ptr<Servo>> ServoInitiate::Initiate()
{
    std::list<std::shared_ptr<Servo>> servos;
    auto servo = std::make_shared<Servo>("01", 1);
    servos.push_back(servo);
    return servos;
}