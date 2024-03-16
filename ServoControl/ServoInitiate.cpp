#include "ServoInitiate.h"

std::list<Servo> ServoInitiate::Initiate()
{
    std::list<Servo> servos;
    servos.push_back(Servo("01", 1));
}