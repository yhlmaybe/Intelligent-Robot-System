#ifndef MSGMANAGER_H
#define MSGMANAGER_H

#include <string>
#include <list>

#include <sstream>
#include <pugixml.hpp>


enum MessageFunction { Default, JointDatas, EndEffectorDatas, BrainDeepLearnFormDatas };

struct ServoMsg
{
    int id;
    std::string name;
    int position;
    double time;
    double velocity;
};

struct JointDatasMsg
{

};


class MsgConvertHandle
{
public:
    static std::string ServoMsgsToString(ServoMsg serveMsg);
    static std::string ServoMsgsToString(std::list<ServoMsg> serveMsgs);
    static ServoMsg StringToServo(std::string msgXML);
    static std::list<ServoMsg> StringToServos(std::string msgXML);
};

#endif