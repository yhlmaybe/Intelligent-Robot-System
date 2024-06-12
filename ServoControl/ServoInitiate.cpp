#include "ServoInitiate.h"

std::list<std::shared_ptr<Servo>> ServoInitiate::Initiate()
{
    std::list<std::shared_ptr<Servo>> servos;
    auto servo = std::make_shared<Servo>("RComp_Thumb_Arth_1", RComp_Thumb_Arth_1);
    servos.push_back(servo);
    return servos;
}