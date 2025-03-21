#include "ServoManager.h"


ServoManager::ServoManager(std::string jointName, std::shared_ptr<Servo> servo)
{
    joint_name = jointName;
    servo_ = servo;
    drive_mode = ServoDriveMode::None;
}

void ServoManager::Initiate(double wide, double high, double fix_x, double fix_y)
{
    wide_ = wide;
    high_ = high;
    fix_x_ = fix_x;
    fix_y_ = fix_y;
    is_initiate = true;
    servo_position_angle = SERVO_INITIAL_POSITION;
    drive_mode = ServoDriveMode::Linear;
}

void ServoManager::Initiate(double radius)
{
    r_ = radius;
    is_initiate = true;
    servo_position_angle = SERVO_INITIAL_POSITION;
    drive_mode = ServoDriveMode::Rotate;
}

void ServoManager::Initiate(JointParam param)
{
    if(param.name == joint_name) 
    {
        if(param.r != 0)
        {
            Initiate(param.r);
        }
        else
        {
            Initiate(param.wide, param.high, param.fix_x, param.fix_y);
        }
    }
    else 
    {
        IRS_MESSAGE(joint_name + " servo initial error, name error");
    }
}

bool ServoManager::SetServoPosition(double servoRotateAngle, double time)
{
    if(!is_initiate)
    {
        IRS_MESSAGE(joint_name + " servo management calculator is not initialized");
        return false;
    }
 
    double servo_cal_angle = servo_position_angle + servoRotateAngle;
    if(servo_cal_angle > SERVO_MAX_POSITION || servo_cal_angle < SERVO_MIN_POSITION)
    {
        IRS_MESSAGE(joint_name + " servo rotation overrun");
        return false;
    }

    servo_->operate->SetServoPosition(static_cast<int>(servo_cal_angle * DEGREE_TO_ROTATE_PARAMETER), time);
    
    servo_position_angle = servo_cal_angle;
    return true;
}

bool ServoManager::CheckServoPosition(double beforeJointPosition, double afterJointPosition)
{
    double rotate_angle = GetServoRotateAngle(beforeJointPosition, afterJointPosition);
    double servo_cal_angle = servo_position_angle + rotate_angle;
    if(servo_cal_angle > SERVO_MAX_POSITION || servo_cal_angle < SERVO_MIN_POSITION)
    {
        IRS_MESSAGE(joint_name + " servo rotation overrun");
        return false;
    }
    return true;
}

bool ServoManager::IsUpdateJointPosBound(double& resMax, double& resMin, double max, double min)
{
    bool is_pass_max = CheckServoPosition(0, max);
    bool is_pass_min = CheckServoPosition(0, min);
    if(is_pass_max && is_pass_min) return false;
    double cal_max = max;
    while(!is_pass_max)
    {
        cal_max = cal_max - 0.05;
        is_pass_max = CheckServoPosition(0, cal_max);
        if(cal_max < 0)
        {
            resMax = 0, resMin = 0;
            return true;
        }
    }
    resMax = cal_max;

    double cal_min = min;
    while(!is_pass_min)
    {
        cal_min = cal_min + 0.05;
        is_pass_min = CheckServoPosition(0, cal_min);
        if(cal_min > 0)
        {
            resMax = 0, resMin = 0;
            return true;
        }
    }
    resMin = cal_min;
    return true;
}

double ServoManager::GetServoRotateAngle(double beforeJointPosition, double afterJointPosition)
{
    double linear_dis;
    if(drive_mode == ServoDriveMode::Linear)
    {
        linear_dis = CalLineDistance(wide_, high_, fix_x_, fix_y_, afterJointPosition) - CalLineDistance(wide_, high_, fix_x_, fix_y_, beforeJointPosition);
    }
    else if(drive_mode == ServoDriveMode::Rotate)
    {
        linear_dis = r_ * (afterJointPosition - beforeJointPosition) * ANGLE_TO_RADIAN;
    }
    else
    {
        linear_dis = 0;
    }
    return linear_dis * SERVO_LINE_TO_ANGLE;
}

void ServoManager::Reset()
{
    servo_position_angle = SERVO_INITIAL_POSITION;
    servo_->operate->SetServoPosition(500, 200/*ms*/);
}

double ServoManager::CalLineDistance(double wide, double high, double fix_x, double fix_y, double angle)
{
    double theta = angle * ANGLE_TO_RADIAN;

    double x = -wide * cos(theta) - high * sin(theta);
    double y = -wide * sin(theta) + high * cos(theta);

    //sqrt( (px - x)^2 + (py - y)^2 )
    double length = sqrt(pow(fix_x - x, 2) + pow(fix_y - y, 2));

    return length;
}

std::map<std::string, std::shared_ptr<ServoManager>> ServoTools::Initiate()
{
    std::map<std::string, std::shared_ptr<ServoManager>> res;
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
    {
        IRS_MESSAGE("get arm joint parameter error");
        return res;
    }
    std::string file = std::string(cwd) + "/Arm_Joint_Cal_Parameter.xml";
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError eResult = doc.LoadFile(file.c_str());

    if (eResult != tinyxml2::XML_SUCCESS) 
    {
        IRS_MESSAGE("Error: Failed to load Arm_Joint_Cal_Parameter file"); 
        return res;
    }

    tinyxml2::XMLElement* root = doc.FirstChildElement("ArmJointConfig");
    if (root == nullptr) 
    {
        IRS_MESSAGE("Error: No <ArmJointConfig> root element found in"); 
        return res;
    }

    std::vector<JointParam> jointParams;
    for (tinyxml2::XMLElement *jointElem = root->FirstChildElement("joint");
         jointElem != nullptr;
         jointElem = jointElem->NextSiblingElement("joint"))
    {
        JointParam jp;

        const char *nameAttr = jointElem->Attribute("name");
        if (nameAttr)
        {
            jp.name = std::string(nameAttr);
        }

        tinyxml2::XMLElement *wideElem = jointElem->FirstChildElement("wide");
        if (wideElem)
        {
            wideElem->QueryDoubleText(&jp.wide);
        }

        tinyxml2::XMLElement *highElem = jointElem->FirstChildElement("high");
        if (highElem)
        {
            highElem->QueryDoubleText(&jp.high);
        }

        tinyxml2::XMLElement *fixxElem = jointElem->FirstChildElement("fix_x");
        if (fixxElem)
        {
            fixxElem->QueryDoubleText(&jp.fix_x);
        }

        tinyxml2::XMLElement *fixyElem = jointElem->FirstChildElement("fix_y");
        if (fixyElem)
        {
            fixyElem->QueryDoubleText(&jp.fix_y);
        }

        std::make_shared<Servo>("RComp_Thumb_Arth_1", RComp_Thumb_Arth_1);
        jointParams.push_back(jp);
    }

    std::map<std::string, std::shared_ptr<Servo>> jointServos;
    jointServos["ThumbArth_1_Joint"] = std::make_shared<Servo>("RComp_Thumb_Arth_1", RComp_Thumb_Arth_1);
    jointServos["ThumbArth_2_Joint"] = std::make_shared<Servo>("RComp_Thumb_Arth_2", RComp_Thumb_Arth_2);
    jointServos["ThumbArth_3_Joint"] = std::make_shared<Servo>("RComp_Thumb_Arth_3", RComp_Thumb_Arth_3);
    jointServos["ThumbArth_4_Joint"] = std::make_shared<Servo>("RComp_Thumb_Arth_4", RComp_Thumb_Arth_4);
    jointServos["IndexFingerArth_1_Joint"] = std::make_shared<Servo>("RComp_IndexFinger_Arth_1", RComp_IndexFinger_Arth_1);
    jointServos["IndexFingerArth_2_Joint"] = std::make_shared<Servo>("RComp_IndexFinger_Arth_2", RComp_IndexFinger_Arth_2);
    jointServos["IndexFingerArth_3_Joint"] = std::make_shared<Servo>("RComp_IndexFinger_Arth_3", RComp_IndexFinger_Arth_3);
    jointServos["IndexFingerArth_4_Joint"] = std::make_shared<Servo>("RComp_IndexFinger_Arth_4", RComp_IndexFinger_Arth_4);
    jointServos["MidFingerArth_1_Joint"] = std::make_shared<Servo>("RComp_MidFinger_Arth_1", RComp_MidFinger_Arth_1);
    jointServos["MidFingerArth_2_Joint"] = std::make_shared<Servo>("RComp_MidFinger_Arth_2", RComp_MidFinger_Arth_2);
    jointServos["MidFingerArth_3_Joint"] = std::make_shared<Servo>("RComp_MidFinger_Arth_3", RComp_MidFinger_Arth_3);
    jointServos["MidFingerArth_4_Joint"] = std::make_shared<Servo>("RComp_MidFinger_Arth_4", RComp_MidFinger_Arth_4);
    jointServos["FourthFingerArth_1_Joint"] = std::make_shared<Servo>("RComp_FourthFinger_Arth_1", RComp_FourthFinger_Arth_1);
    jointServos["FourthFingerArth_2_Joint"] = std::make_shared<Servo>("RComp_FourthFinger_Arth_2", RComp_FourthFinger_Arth_2);
    jointServos["FourthFingerArth_3_Joint"] = std::make_shared<Servo>("RComp_FourthFinger_Arth_3", RComp_FourthFinger_Arth_3);
    jointServos["FourthFingerArth_4_Joint"] = std::make_shared<Servo>("RComp_FourthFinger_Arth_4", RComp_FourthFinger_Arth_4);
    jointServos["LittleFingerArth_1_Joint"] = std::make_shared<Servo>("RComp_LittleFinger_Arth_1", RComp_LittleFinger_Arth_1);
    jointServos["LittleFingerArth_2_Joint"] = std::make_shared<Servo>("RComp_LittleFinger_Arth_2", RComp_LittleFinger_Arth_2);
    jointServos["LittleFingerArth_3_Joint"] = std::make_shared<Servo>("RComp_LittleFinger_Arth_3", RComp_LittleFinger_Arth_3);
    jointServos["LittleFingerArth_4_Joint"] = std::make_shared<Servo>("RComp_LittleFinger_Arth_4", RComp_LittleFinger_Arth_4);
    jointServos["Palm_Joint"] = std::make_shared<Servo>("RComp_Palm", RComp_Palm);
    jointServos["WristArth_Joint"] = std::make_shared<Servo>("RComp_Wrist_Arth", RComp_Wrist_Arth);

    for (size_t i = 0; i < jointParams.size(); ++i)
    {
        std::string name = jointParams[i].name;
        auto it = jointServos.find(name);
        if (it != jointServos.end())
        {
            std::shared_ptr<ServoManager> serManager = std::make_shared<ServoManager>(name, it->second);
            serManager->Initiate(jointParams[i]);
            res[name] = serManager;
        }
        else IRS_MESSAGE(name + " failed to initialize the name list");
    }

    return res;
}

std::vector<bool> ServoTools::SetServoPositions(std::vector<std::shared_ptr<ServoManager>> servoManagers, std::vector<double> beforeJointPosition, std::vector<double> afterJointPosition, double time)
{
    std::vector<bool> res;
    if(servoManagers.size() != beforeJointPosition.size() || beforeJointPosition.size() != afterJointPosition.size())
    {
        IRS_MESSAGE("when setting servo positions, the number of servo is not equal to the number of angles");
        return res;
    } 

    double cal_time = time;
    std::vector<double> rotate_angles;
    for(size_t i = 0; i < servoManagers.size(); ++i)
    {
        double rotate_angle = servoManagers[i]->GetServoRotateAngle(beforeJointPosition[i], afterJointPosition[i]);
        double rotate_time = rotate_angle / SERVO_MAX_ANGLEVELOCITY;
        if(rotate_time > cal_time) cal_time = rotate_time;
        rotate_angles.push_back(rotate_angle);
    }

    for(size_t i = 0; i < servoManagers.size(); ++i)
    {
        res.push_back(servoManagers[i]->SetServoPosition(rotate_angles[i], cal_time));
    }
    return res;
}

bool ServoTools::SetServoPosition(std::shared_ptr<ServoManager> servoManager, double beforeJointPosition, double afterJointPosition, double time)
{
    std::vector<std::shared_ptr<ServoManager>> managers = {servoManager};
    std::vector<double> before_pos = {beforeJointPosition};
    std::vector<double> after_pos = {afterJointPosition};
    bool res = SetServoPositions(managers, before_pos, after_pos, time)[0];
    return res;
}

void ServoTools::ResetServo(std::vector<std::shared_ptr<ServoManager>> servoManagers)
{
    for(std::shared_ptr<ServoManager> it : servoManagers)
    {
        it->Reset();
    }
}