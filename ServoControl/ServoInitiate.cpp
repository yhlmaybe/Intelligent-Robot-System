#include "ServoInitiate.h"

std::list<std::shared_ptr<Servo>> ServoInitiate::Initiate()
{
    std::list<std::shared_ptr<Servo>> servos;
    servos.push_back(std::make_shared<Servo>("RComp_Thumb_Arth_1", RComp_Thumb_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_Thumb_Arth_2", RComp_Thumb_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_Thumb_Arth_3", RComp_Thumb_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_Thumb_Arth_4", RComp_Thumb_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_IndexFinger_Arth_1", RComp_IndexFinger_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_IndexFinger_Arth_2", RComp_IndexFinger_Arth_2));
    servos.push_back(std::make_shared<Servo>("RComp_IndexFinger_Arth_3", RComp_IndexFinger_Arth_3));
    servos.push_back(std::make_shared<Servo>("RComp_IndexFinger_Arth_4", RComp_IndexFinger_Arth_4));
    servos.push_back(std::make_shared<Servo>("RComp_MidFinger_Arth_1", RComp_MidFinger_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_MidFinger_Arth_2", RComp_MidFinger_Arth_2));
    servos.push_back(std::make_shared<Servo>("RComp_MidFinger_Arth_3", RComp_MidFinger_Arth_3));
    servos.push_back(std::make_shared<Servo>("RComp_MidFinger_Arth_4", RComp_MidFinger_Arth_4));
    servos.push_back(std::make_shared<Servo>("RComp_FourthFinger_Arth_1", RComp_FourthFinger_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_FourthFinger_Arth_2", RComp_FourthFinger_Arth_2));
    servos.push_back(std::make_shared<Servo>("RComp_FourthFinger_Arth_3", RComp_FourthFinger_Arth_3));
    servos.push_back(std::make_shared<Servo>("RComp_FourthFinger_Arth_4", RComp_FourthFinger_Arth_4));
    servos.push_back(std::make_shared<Servo>("RComp_LittleFinger_Arth_1", RComp_LittleFinger_Arth_1));
    servos.push_back(std::make_shared<Servo>("RComp_LittleFinger_Arth_2", RComp_LittleFinger_Arth_2));
    servos.push_back(std::make_shared<Servo>("RComp_LittleFinger_Arth_3", RComp_LittleFinger_Arth_3));
    servos.push_back(std::make_shared<Servo>("RComp_LittleFinger_Arth_4", RComp_LittleFinger_Arth_4));
    servos.push_back(std::make_shared<Servo>("RComp_Wrist_Arth_YZ", RComp_Wrist_Arth_YZ));
    servos.push_back(std::make_shared<Servo>("RComp_Wrist_Arth_XY", RComp_Wrist_Arth_XY));

    for(auto it = servos.begin(); it != servos.end(); ++it)
    {
        it->get()->operate->SetServoPosition(0, 500);
    }
    return servos;
}