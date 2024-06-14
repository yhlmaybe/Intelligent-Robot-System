#include "ServoInitiate.h"

std::list<std::shared_ptr<Servo>> ServoInitiate::Initiate()
{
    std::list<std::shared_ptr<Servo>> servos;
    auto servo = std::make_shared<Servo>("RComp_Wrist_Arth_XY", RComp_Wrist_Arth_XY);
    servos.push_back(servo);

    std::list<std::future<void>> results;
    for(auto it = servos.begin(); it != servos.end(); ++it)
    {
        results.push_back(std::async(std::launch::async, [&it](){
            it->get()->operate->SetServoPosition(500, 1);//range 0-1000 , it make servo's degree mid
        }));
    }

    for(auto& resule : results)
    {
        resule.wait();
    }

    return servos;
}