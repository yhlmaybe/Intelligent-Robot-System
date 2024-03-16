#include <string>
#include <list>

#include <sstream>
#include <pugixml.hpp>


struct ServoMsg
{
    int id;
    std::string name;
    int position;
};


class MsgConvertHandle
{
public:
    static std::string ServoMsgsToString(ServoMsg serveMsg);
    static std::string ServoMsgsToString(std::list<ServoMsg> serveMsgs);
    static ServoMsg StringToServo(std::string msgXML);
    static std::list<ServoMsg> StringToServos(std::string msgXML);
};