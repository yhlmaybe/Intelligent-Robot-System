#include "MsgManager.h"

std::string MsgConvertHandle::ServoMsgsToString(ServoMsg servoMsg)
{
    std::list<ServoMsg> servoMsgs;
    servoMsgs.push_back(servoMsg);
    return ServoMsgsToString(servoMsgs);
}

std::string MsgConvertHandle::ServoMsgsToString(std::list<ServoMsg> servoMsgs)
{
    pugi::xml_document doc;
    auto dateNode = doc.append_child("ServeMsg");
    for(const ServoMsg& msg : servoMsgs)
    {
        auto item = dateNode.append_child("Item");
        item.append_attribute("id").set_value(msg.id);
        item.append_attribute("name").set_value(msg.name.c_str());
        item.append_attribute("position").set_value(msg.position);
    }
    std::stringstream msgString;
    doc.save(msgString, "  ");
    return msgString.str();
}

ServoMsg MsgConvertHandle::StringToServo(std::string msgXML)
{
    return StringToServos(msgXML).front();
}

std::list<ServoMsg> MsgConvertHandle::StringToServos(std::string msgXML)
{
    std::list<ServoMsg> msgList;
    pugi::xml_document parsedDoc;
    parsedDoc.load_string(msgXML.c_str());
    for(pugi::xml_node itemNode = parsedDoc.child("ServeMsg").child("Item"); itemNode; itemNode = itemNode.next_sibling("Item"))
    {
        ServoMsg servoMsg;
        servoMsg.id = itemNode.attribute("id").as_int();
        servoMsg.name = itemNode.attribute("name").as_string();
        servoMsg.position = itemNode.attribute("position").as_int();
        msgList.push_back(servoMsg);
    }
    return msgList;
}