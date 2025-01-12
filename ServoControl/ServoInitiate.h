#ifndef SERVOINITIATE_H
#define SERVOINITIATE_H

#include "SCDrive.h"
#include "ServoID.h"

class ServoInitiate
{
public:
    static std::list<std::shared_ptr<Servo>> Initiate();
};

#endif